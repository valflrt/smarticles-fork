use std::{fmt::Display, sync::mpsc::Sender, time::Duration};

use eframe::egui::Vec2;

use crate::{ai::net::Network, mat::Mat2D, CLASS_COUNT};

#[derive(Debug, Clone)]
pub enum Event {
    Exit,

    SpawnParticles,

    SimulationStart,
    SimulationPause,
    SimulationReset,

    NetworkStart,
    NetworkStop,

    StartTraining(usize),
    TrainingStopped,
    EvaluateNetworks,

    StateUpdate(StateUpdate),
}

impl Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Event::Exit => write!(f, "Exit"),
            Event::SpawnParticles => write!(f, "SpawnParticles"),
            Event::SimulationStart => write!(f, "SimulationStart"),
            Event::SimulationPause => write!(f, "SimulationPause"),
            Event::SimulationReset => write!(f, "SimulationReset"),
            Event::NetworkStart => write!(f, "NetworkStart"),
            Event::NetworkStop => write!(f, "NetworkStop"),
            Event::StartTraining(_) => write!(f, "StartTraining"),
            Event::TrainingStopped => write!(f, "TrainingStopped"),
            Event::EvaluateNetworks => write!(f, "EvaluateNetworks"),
            Event::StateUpdate(state_update) => write!(f, "{}", state_update),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Recipient {
    App,
    Sim,
    Training,
    _AppAndSim,
    _AppAndTraining,
    SimAndTraining,
}

#[derive(Debug, Clone)]
pub struct Senders {
    origin: Option<Recipient>,

    pub app_sender: Sender<Event>,
    pub sim_sender: Sender<Event>,
    pub training_sender: Sender<Event>,
}

impl Senders {
    pub fn new(
        app_sender: Sender<Event>,
        sim_sender: Sender<Event>,
        training_sender: Sender<Event>,
    ) -> Self {
        Senders {
            origin: None,

            app_sender,
            sim_sender,
            training_sender,
        }
    }

    pub fn origin(mut self, origin: Recipient) -> Self {
        self.origin = Some(origin);
        self
    }

    pub fn send(&self, send_to: Recipient, event: Event) {
        println!(
            "{}to {:?}: {}",
            self.origin
                .clone()
                .map_or(String::new(), |origin| format!("{:?} ", origin)),
            send_to,
            event
        );
        match send_to {
            Recipient::App => self.app_sender.send(event).unwrap(),
            Recipient::Sim => self.sim_sender.send(event).unwrap(),
            Recipient::Training => self.training_sender.send(event).unwrap(),
            Recipient::_AppAndSim => {
                self.app_sender.send(event.clone()).unwrap();
                self.sim_sender.send(event.clone()).unwrap();
            }
            Recipient::_AppAndTraining => {
                self.app_sender.send(event.clone()).unwrap();
                self.training_sender.send(event).unwrap();
            }
            Recipient::SimAndTraining => {
                self.sim_sender.send(event.clone()).unwrap();
                self.training_sender.send(event).unwrap();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct StateUpdate {
    pub particle_positions: Option<Mat2D<Vec2>>,
    pub computation_duration: Option<Duration>,

    pub power_matrix: Option<Mat2D<i8>>,
    pub particle_counts: Option<[usize; CLASS_COUNT]>,

    pub training_generation: Option<usize>,
    pub network_ranking: Option<Vec<(f32, Network)>>,
    pub inference_network: Option<Network>,
    pub target_angle: Option<f32>,
}

impl StateUpdate {
    pub fn new() -> Self {
        StateUpdate {
            particle_positions: None,
            computation_duration: None,
            power_matrix: None,
            particle_counts: None,
            training_generation: None,
            network_ranking: None,
            inference_network: None,
            target_angle: None,
        }
    }

    pub fn particle_positions(mut self, particle_positions: &Mat2D<Vec2>) -> StateUpdate {
        self.particle_positions = Some(particle_positions.clone());
        self
    }
    pub fn computation_duration(mut self, computation_duration: Duration) -> StateUpdate {
        self.computation_duration = Some(computation_duration);
        self
    }
    pub fn power_matrix(mut self, power_matrix: &Mat2D<i8>) -> StateUpdate {
        self.power_matrix = Some(power_matrix.clone());
        self
    }
    pub fn particle_counts(mut self, particle_counts: &[usize; 4]) -> StateUpdate {
        self.particle_counts = Some(particle_counts.clone());
        self
    }
    pub fn training_generation(mut self, training_generation: usize) -> StateUpdate {
        self.training_generation = Some(training_generation);
        self
    }
    pub fn network_ranking(mut self, network_ranking: &Vec<(f32, Network)>) -> StateUpdate {
        self.network_ranking = Some(network_ranking.clone());
        self
    }
    pub fn inference_network(mut self, inference_network: &Network) -> StateUpdate {
        self.inference_network = Some(inference_network.clone());
        self
    }
    pub fn target_angle(mut self, target_angle: f32) -> StateUpdate {
        self.target_angle = Some(target_angle);
        self
    }
}

impl Display for StateUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut fields = Vec::new();

        if self.particle_positions.is_some() {
            fields.push("particle_positions");
        }
        if self.computation_duration.is_some() {
            fields.push("computation_duration");
        }
        if self.power_matrix.is_some() {
            fields.push("power_matrix");
        }
        if self.particle_counts.is_some() {
            fields.push("particle_counts");
        }
        if self.training_generation.is_some() {
            fields.push("training_generation");
        }
        if self.network_ranking.is_some() {
            fields.push("network_ranking");
        }
        if self.inference_network.is_some() {
            fields.push("inference_network");
        }
        if self.target_angle.is_some() {
            fields.push("target_angle");
        }

        write!(f, "StateUpdate {{ {} }}", fields.join(", "))
    }
}
