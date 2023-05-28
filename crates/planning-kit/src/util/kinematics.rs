use std::f64::consts::PI;

use nalgebra::{Isometry3, Translation3, Unit, UnitQuaternion, Vector3};
use parry3d_f64::shape::{Ball, Compound, Cuboid, Cylinder, SharedShape};
use petgraph::{
    algo::toposort,
    graph::{EdgeReferences, NodeReferences},
    prelude::{DiGraph, EdgeIndex, NodeIndex},
    visit::{EdgeRef, IntoNodeReferences},
    Direction,
};

#[derive(Clone)]
pub struct KinematicChain {
    chain: DiGraph<Link, Joint>,
}

impl KinematicChain {
    pub fn from_urdf(robot: &urdf_rs::Robot) -> Self {
        let chain = parse_urdf(robot);

        Self { chain }
    }

    pub fn link(&self, ix: NodeIndex) -> &Link {
        &self.chain[ix]
    }

    pub fn links(&self) -> NodeReferences<Link> {
        self.chain.node_references()
    }

    pub fn joint(&self, ix: EdgeIndex) -> &Joint {
        &self.chain[ix]
    }

    pub fn joints(&self) -> EdgeReferences<Joint> {
        self.chain.edge_references()
    }

    pub fn nq(&self) -> usize {
        self.chain.edge_weights().map(|j| j.typ.nq()).sum()
    }

    pub fn add_link(&mut self, link: Link) -> NodeIndex {
        self.chain.add_node(link)
    }

    pub fn add_joint(
        &mut self,
        parent_link_ix: NodeIndex,
        child_link_ix: NodeIndex,
        joint: Joint,
    ) -> EdgeIndex {
        assert!(self.parent_joint(parent_link_ix).is_none());
        self.chain.add_edge(parent_link_ix, child_link_ix, joint)
    }

    fn parent_joint(&self, link_ix: NodeIndex) -> Option<EdgeIndex> {
        // Each link will have at most a single incoming (parent) joint by
        // construction of the graph.
        self.chain
            .edges_directed(link_ix, Direction::Incoming)
            .next()
            .map(|edge| edge.id())
    }

    pub fn forward_kinematics(&self, q: &[f64]) -> ForwardKinematics {
        let mut qidx = 0;
        let mut tfms = vec![Isometry3::default(); self.chain.edge_count()];

        // SAFETY: already guaranteed loops are not present at construction
        // time, so unwrapping the toposort is safe.
        for link_ix in toposort(&self.chain, None).unwrap() {
            let parent_joint = self.parent_joint(link_ix);
            let parent_joint_tfm = parent_joint
                .map(|j_ix| tfms[j_ix.index()])
                .unwrap_or_default();

            // Compute the world pose of each child joint of this link.
            for edge in self.chain.edges_directed(link_ix, Direction::Outgoing) {
                let child_joint = edge.weight();
                let child_joint_q = &q[qidx..(qidx + child_joint.typ.nq())];
                let child_joint_local_tfm = child_joint.typ.forward_kinematics(child_joint_q);
                let child_joint_tfm = parent_joint_tfm * child_joint.origin * child_joint_local_tfm;

                tfms[edge.id().index()] = child_joint_tfm;
                qidx += child_joint.typ.nq();
            }
        }

        ForwardKinematics {
            chain: self,
            joint_tfms: tfms,
        }
    }
}

pub struct ForwardKinematics<'a> {
    chain: &'a KinematicChain,
    joint_tfms: Vec<Isometry3<f64>>,
}

impl<'a> ForwardKinematics<'a> {
    pub fn joint_tfm(&self, joint_ix: EdgeIndex) -> Isometry3<f64> {
        self.joint_tfms[joint_ix.index()]
    }

    pub fn link_tfm(&self, link_ix: NodeIndex) -> Isometry3<f64> {
        if let Some(parent_joint) = self.chain.parent_joint(link_ix) {
            self.joint_tfms[parent_joint.index()]
        } else {
            // TODO: None?
            Isometry3::identity()
        }
    }
}

#[derive(Clone)]
pub struct Joint {
    pub name: String,
    pub typ: JointType,
    pub origin: Isometry3<f64>,
}

#[derive(Clone, Debug)]
pub enum JointType {
    Revolute(Unit<Vector3<f64>>),
    Prismatic(Unit<Vector3<f64>>),
    Fixed,
}

impl JointType {
    pub fn nq(&self) -> usize {
        match self {
            JointType::Revolute(_) => 1,
            JointType::Prismatic(_) => 1,
            JointType::Fixed => 0,
        }
    }

    pub fn forward_kinematics(&self, q: &[f64]) -> Isometry3<f64> {
        match self {
            JointType::Revolute(axis) => {
                let rotation = UnitQuaternion::from_axis_angle(axis, q[0]);
                Isometry3::from_parts(Translation3::identity(), rotation)
            }
            JointType::Prismatic(axis) => {
                let translation = axis.scale(q[0]);
                Isometry3::from_parts(Translation3::from(translation), UnitQuaternion::identity())
            }
            JointType::Fixed => Isometry3::identity(),
        }
    }
}

#[derive(Clone)]
pub struct Link {
    pub name: String,
    pub collision: Option<Compound>,
}

fn urdf_to_tfm(pose: &urdf_rs::Pose) -> Isometry3<f64> {
    let xyz = Vector3::from_row_slice(&pose.xyz.0);
    let rot = UnitQuaternion::from_euler_angles(pose.rpy.0[0], pose.rpy.0[1], pose.rpy.0[2]);
    Isometry3::from_parts(nalgebra::Translation::from(xyz), rot)
}

fn parse_geometry(collision: &[urdf_rs::Collision]) -> Option<Compound> {
    let mut shapes = vec![];
    for c in collision {
        let geometry_origin = urdf_to_tfm(&c.origin);
        match &c.geometry {
            urdf_rs::Geometry::Box { size } => {
                let half_extents = Vector3::from_row_slice(&size.0) / 2.0;
                shapes.push((geometry_origin, SharedShape::new(Cuboid::new(half_extents))))
            }
            urdf_rs::Geometry::Cylinder { radius, length } => {
                let tfm = geometry_origin
                    * Isometry3::from_parts(
                        Translation3::identity(),
                        UnitQuaternion::from_euler_angles(PI / 2.0, 0.0, 0.0),
                    );
                shapes.push((
                    tfm,
                    SharedShape::new(Cylinder {
                        half_height: length / 2.0,
                        radius: *radius,
                    }),
                ));
            }
            urdf_rs::Geometry::Sphere { radius } => {
                shapes.push((geometry_origin, SharedShape::new(Ball::new(*radius))))
            }
            _ => {
                todo!("TODO: collision shape not supported: {:?}", &c.geometry);
            }
        }
    }

    if !shapes.is_empty() {
        Some(parry3d_f64::shape::Compound::new(shapes))
    } else {
        None
    }
}

fn parse_urdf(urdf: &urdf_rs::Robot) -> DiGraph<Link, Joint> {
    let mut graph = DiGraph::<Link, Joint>::new();

    for link in &urdf.links {
        let collision = parse_geometry(&link.collision);
        graph.add_node(Link {
            name: link.name.clone(),
            collision,
        });
    }

    for joint in &urdf.joints {
        let (parent_ix, _) = graph
            .node_references()
            .find(|l| l.1.name == joint.parent.link)
            .unwrap_or_else(|| panic!("joint parent link '{}' does not exist", joint.parent.link));
        let (child_ix, _) = graph
            .node_references()
            .find(|l| l.1.name == joint.child.link)
            .unwrap_or_else(|| panic!("joint child link '{}' does not exist", joint.parent.link));

        let joint_type = match &joint.joint_type {
            urdf_rs::JointType::Revolute => JointType::Revolute(Unit::new_normalize(
                Vector3::from_row_slice(&joint.axis.xyz.0),
            )),
            urdf_rs::JointType::Prismatic => JointType::Prismatic(Unit::new_normalize(
                Vector3::from_row_slice(&joint.axis.xyz.0),
            )),
            urdf_rs::JointType::Fixed => JointType::Fixed,
            x => panic!("joint type not supported: {:?}", x),
        };

        let origin = urdf_to_tfm(&joint.origin);
        graph.add_edge(
            parent_ix,
            child_ix,
            Joint {
                name: joint.name.clone(),
                typ: joint_type,
                origin,
            },
        );
    }

    graph
}
