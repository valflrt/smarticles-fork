mod ai;
mod app;
mod mat;
mod simulation;

use std::{
    fs,
    path::{Path, PathBuf},
    sync::mpsc::{channel, Sender},
    thread,
    time::Duration,
};

use ai::{
    batch::Batch,
    net::{ActivationFn, Layer, Network},
    training::{BATCH_SIZE, NETWORK_INPUT_SIZE, NETWORK_OUTPUT_SIZE},
};
use app::{
    simulation_manager::SimulationManager, training_manager::TrainingManager, SmarticlesApp,
};
use eframe::{
    egui::{Vec2, ViewportBuilder},
    epaint::{Color32, Hsva},
    NativeOptions,
};
use mat::Mat2D;
use postcard::from_bytes;

const CLASS_COUNT: usize = 4;

/// Min particle count
const MIN_PARTICLE_COUNT: usize = 0;
/// Max particle count per class
const MAX_PARTICLE_COUNT: usize = 20000;
/// When randomizing particle counts, this is the lowest
/// possible value, this prevent random particle counts from
/// being under this value
const RANDOM_MIN_PARTICLE_COUNT: usize = 200;
/// When randomizing particle counts, this is the highest
/// possible value, this prevent random particle counts from
/// being above this value
const RANDOM_MAX_PARTICLE_COUNT: usize = MAX_PARTICLE_COUNT;

const MAX_POWER: i8 = 100;
const MIN_POWER: i8 = -MAX_POWER;

fn main() {
    let batch = if Path::new("./batches/batch_gen_0").exists() {
        let mut batch_gen = 0;
        let mut batch_path = Path::new("./batches/batch_gen_0").to_path_buf();

        const GAP: usize = 50;
        let mut gap = GAP;

        while gap > 0 {
            let path: PathBuf = format!("./batches/batch_gen_{}", batch_gen).into();
            if path.exists() {
                gap = GAP;
                batch_path = path;
            } else {
                gap -= 1;
            }
            batch_gen += 1;
        }
        from_bytes::<Batch>(&fs::read(batch_path).unwrap()).unwrap()
    } else {
        const HIDDEN_LAYER_SIZE: usize = 8;
        let batch = Batch::new(Vec::from_iter((0..BATCH_SIZE).map(|_| {
            Network::new([
                Layer::random(NETWORK_INPUT_SIZE, HIDDEN_LAYER_SIZE, ActivationFn::Tanh),
                Layer::random(HIDDEN_LAYER_SIZE, NETWORK_OUTPUT_SIZE, ActivationFn::Tanh),
            ])
        })));
        batch.save();
        batch
    };

    let options = NativeOptions {
        viewport: ViewportBuilder::default().with_fullscreen(true),
        ..Default::default()
    };

    let (ui_sender, ui_receiver) = channel::<SmarticlesEvent>();
    let (sim_sender, sim_receiver) = channel::<SmarticlesEvent>();
    let (training_sender, training_receiver) = channel::<SmarticlesEvent>();

    let senders = Senders::new(ui_sender, sim_sender, training_sender);

    eframe::run_native(
        "Smarticles",
        options,
        Box::new(|cc| {
            let frame = cc.egui_ctx.clone();

            let generation = batch.generation;

            let senders_clone = senders.clone();
            let training_handle = thread::spawn(move || {
                let mut training_manager =
                    TrainingManager::new(batch, senders_clone, training_receiver);

                loop {
                    if !training_manager.update() {
                        break;
                    };
                }
            });

            let senders_clone = senders.clone();
            let simulation_handle = thread::spawn(move || {
                let mut sim_manager = SimulationManager::new(senders_clone, sim_receiver);

                thread::sleep(Duration::from_millis(500));

                loop {
                    if !sim_manager.update() {
                        break;
                    };
                    frame.request_repaint();
                }
            });

            Ok(Box::new(SmarticlesApp::new(
                ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"]
                    .iter()
                    .take(CLASS_COUNT)
                    .copied()
                    .enumerate()
                    .map(|(i, class_name)| {
                        let [r, g, b] = Hsva::new(
                            // tinkering to make purple appear first. I like purple.
                            (((CLASS_COUNT - i - 1) % CLASS_COUNT) as f32) / (CLASS_COUNT as f32),
                            0.9,
                            0.9,
                            1.,
                        )
                        .to_srgb();
                        (class_name, Color32::from_rgb(r, g, b))
                    })
                    .collect::<Vec<(&str, Color32)>>()
                    .try_into()
                    .unwrap(),
                generation,
                senders,
                ui_receiver,
                Some(simulation_handle),
                Some(training_handle),
            )))
        }),
    )
    .unwrap();
}

#[derive(Debug, Clone)]
struct Senders {
    app_sender: Sender<SmarticlesEvent>,
    sim_sender: Sender<SmarticlesEvent>,
    training_sender: Sender<SmarticlesEvent>,
}

impl Senders {
    pub fn new(
        ui_sender: Sender<SmarticlesEvent>,
        sim_sender: Sender<SmarticlesEvent>,
        training_sender: Sender<SmarticlesEvent>,
    ) -> Self {
        Senders {
            app_sender: ui_sender,
            sim_sender,
            training_sender,
        }
    }

    pub fn send_to_app(&self, event: SmarticlesEvent) {
        self.app_sender.send(event).unwrap()
    }
    pub fn send_to_simulation_manager(&self, event: SmarticlesEvent) {
        self.sim_sender.send(event).unwrap()
    }
    pub fn send_to_training_manager(&self, event: SmarticlesEvent) {
        self.training_sender.send(event).unwrap()
    }
}

#[derive(Debug, Clone)]
enum SmarticlesEvent {
    Quit,

    SpawnParticles,

    /// Particle positions and elapsed time (if available).
    SimulationResults(Mat2D<Vec2>, Option<Duration>),
    _SimulationInfo(),

    PowerMatrixChange(Mat2D<i8>),
    ParticleCountsUpdate([usize; CLASS_COUNT]),

    SimulationStart,
    SimulationPause,
    SimulationReset,

    NetworkStart,
    NetworkStop,

    StartTraining(usize),
    GenerationChange(usize),
    NetworkRanking(Vec<(f32, Network)>),
    EvaluateNetworks,

    InferenceNetworkChange(Network),
    TargetAngleChange(f32),
}
