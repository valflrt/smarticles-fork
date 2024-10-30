pub mod simulation_manager;
pub mod training_manager;
mod ui;

use std::{
    collections::{hash_map::DefaultHasher, VecDeque},
    hash::{Hash, Hasher},
    sync::mpsc::Receiver,
    thread::JoinHandle,
};

use byteorder::{ReadBytesExt, WriteBytesExt};
use eframe::{
    egui::{
        Align2, Area, CentralPanel, ComboBox, Context, PointerButton, ScrollArea, Sense, SidePanel,
        Slider, Vec2,
    },
    epaint::Color32,
    App, Frame,
};
use egui_plot::{Line, Plot, PlotPoints};
use rand::{distributions::Open01, rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use simulation_manager::{NetworkState, SimulationState};
use ui::DirectionKnob;

use crate::{
    ai::{net::Network, training::random_target_angle},
    events::{Recipient, StateUpdate},
    mat::Mat2D,
    Event, Senders, CLASS_COUNT, MAX_PARTICLE_COUNT, MAX_POWER, MIN_PARTICLE_COUNT, MIN_POWER,
    RANDOM_MAX_PARTICLE_COUNT, RANDOM_MIN_PARTICLE_COUNT,
};

/// Display diameter of the particles in the simulation (in
/// pixels).
const PARTICLE_DIAMETER: f32 = 0.6;

const DEFAULT_ZOOM: f32 = 2.;
const MIN_ZOOM: f32 = 0.5;
const MAX_ZOOM: f32 = 30.;
const ZOOM_FACTOR: f32 = 1.05;

const MAX_HISTORY_LEN: usize = 10;

#[derive(Debug, PartialEq)]
enum TrainingState {
    Training { target_generation: usize },
    Evaluating,
    Stopped,
}
impl TrainingState {
    pub fn is_running(&self) -> bool {
        matches!(
            self,
            TrainingState::Training { .. } | TrainingState::Evaluating
        )
    }
}

pub struct View {
    zoom: f32,
    pos: Vec2,
    dragging: bool,
    drag_start_pos: Vec2,
    drag_start_view_pos: Vec2,
}

impl View {
    const DEFAULT: View = Self {
        zoom: DEFAULT_ZOOM,
        pos: Vec2::ZERO,
        dragging: false,
        drag_start_pos: Vec2::ZERO,
        drag_start_view_pos: Vec2::ZERO,
    };
}

#[derive(Debug)]
struct ClassProps {
    name: String,
    heading: String,
    color: Color32,
    enabled: bool,
}

pub struct SmarticlesApp {
    classes: [ClassProps; CLASS_COUNT],

    seed: String,

    show_ui: bool,

    view: View,

    selected_particle: (usize, usize),
    follow_selected_particle: bool,

    history: VecDeque<String>,
    selected_history_entry: usize,

    computation_time_graph: VecDeque<f32>,

    particle_counts: [usize; CLASS_COUNT],
    locked_particle_counts: bool,
    particle_positions: Mat2D<Vec2>,
    power_matrix: Mat2D<i8>,
    simulation_state: SimulationState,

    network_state: NetworkState,
    network_ranking: Option<Vec<(f32, Network)>>,
    selected_network: usize,
    target_angle: f32,
    training_state: TrainingState,
    generation: usize,

    senders: Senders,
    receiver: Receiver<Event>,

    simulation_handle: Option<JoinHandle<()>>,
    training_handle: Option<JoinHandle<()>>,

    words: Vec<String>,
}

impl SmarticlesApp {
    pub fn new<S>(
        classes: [(S, Color32); CLASS_COUNT],
        generation: usize,
        senders: Senders,
        receiver: Receiver<Event>,
        simulation_handle: Option<JoinHandle<()>>,
        training_handle: Option<JoinHandle<()>>,
    ) -> Self
    where
        S: ToString,
    {
        let words = include_str!("app/words.txt");
        let words: Vec<String> = words
            .par_lines()
            .filter_map(|w| {
                if w.len() > 8 {
                    return None;
                }
                for chr in w.chars() {
                    if !chr.is_ascii_alphabetic() || chr.is_ascii_uppercase() {
                        return None;
                    }
                }
                Some(w.to_string())
            })
            .collect();

        let target_angle = random_target_angle();
        senders.send(
            Recipient::Sim,
            Event::StateUpdate(StateUpdate::new().target_angle(target_angle)),
        );

        Self {
            seed: "".to_string(),

            classes: classes.map(|(name, color)| ClassProps {
                name: name.to_string(),
                heading: "class ".to_string() + &name.to_string(),
                color,
                enabled: true,
            }),

            show_ui: true,

            view: View::DEFAULT,

            selected_particle: (0, 0),
            follow_selected_particle: false,

            history: VecDeque::new(),
            selected_history_entry: 0,

            computation_time_graph: VecDeque::new(),

            particle_counts: [0; CLASS_COUNT],
            locked_particle_counts: false,
            particle_positions: Mat2D::filled_with(Vec2::ZERO, CLASS_COUNT, MAX_PARTICLE_COUNT),
            power_matrix: Mat2D::filled_with(0, CLASS_COUNT, CLASS_COUNT),
            simulation_state: SimulationState::Paused,

            network_state: NetworkState::Stopped,
            network_ranking: None,
            selected_network: 0,
            target_angle,
            training_state: TrainingState::Stopped,
            generation,

            senders,
            receiver,

            simulation_handle,
            training_handle,

            words,
        }
    }

    fn apply_seed(&mut self) {
        let mut rand = if self.seed.is_empty() {
            SmallRng::from_entropy()
        } else {
            if self.seed.starts_with('@') {
                if let Ok(bytes) = base64::decode(&self.seed[1..]) {
                    self.apply_custom_seed(&bytes);
                    return;
                }
            }
            let mut hasher = DefaultHasher::new();
            self.seed.hash(&mut hasher);
            SmallRng::seed_from_u64(hasher.finish())
        };
        let mut rand = |min: f32, max: f32| min + (max - min) * rand.sample::<f32, _>(Open01);

        const POW_F: f32 = 1.25;

        for i in 0..CLASS_COUNT {
            if !self.locked_particle_counts {
                self.particle_counts[i] = rand(
                    RANDOM_MIN_PARTICLE_COUNT as f32,
                    RANDOM_MAX_PARTICLE_COUNT as f32,
                ) as usize;
            }
            for j in 0..CLASS_COUNT {
                let pow = rand(MIN_POWER as f32, MAX_POWER as f32);
                self.power_matrix[(i, j)] = (pow.signum() * pow.abs().powf(1. / POW_F)) as i8;
            }
        }
    }
    fn export_custom_seed(&self) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        self.power_matrix
            .iter()
            .copied()
            .for_each(|power| bytes.write_i8(power).unwrap());

        format!("@{}", base64::encode(bytes))
    }
    fn apply_custom_seed(&mut self, mut bytes: &[u8]) {
        for i in 0..CLASS_COUNT {
            for j in 0..CLASS_COUNT {
                self.power_matrix[(i, j)] = bytes.read_i8().unwrap_or(0);
            }
        }
    }

    fn update_history(&mut self) {
        if self
            .history
            .front()
            .and_then(|front| Some(&self.seed != front))
            .unwrap_or(true)
        {
            self.history.push_front(self.seed.to_owned());
            if self.history.len() > MAX_HISTORY_LEN {
                self.history.pop_back();
            }
        }
        self.selected_history_entry = 0;
    }

    fn play(&mut self) {
        self.simulation_state = SimulationState::Running;
        self.senders.send(Recipient::Sim, Event::SimulationStart);
    }
    fn pause(&mut self) {
        self.simulation_state = SimulationState::Paused;
        self.senders.send(Recipient::Sim, Event::SimulationPause);
    }
    fn reset(&mut self) {
        self.simulation_state = SimulationState::Paused;
        self.locked_particle_counts = false;
        self.particle_counts = [0; CLASS_COUNT];
        for i in 0..CLASS_COUNT {
            for j in 0..CLASS_COUNT {
                self.power_matrix[(i, j)] = 0;
            }
        }
        self.senders.send(Recipient::Sim, Event::SimulationReset);
    }
}

impl App for SmarticlesApp {
    fn update(&mut self, ctx: &Context, _: &mut Frame) {
        let events = self.receiver.try_iter();
        for event in events {
            match event {
                Event::StateUpdate(StateUpdate {
                    particle_positions,
                    computation_time,

                    power_matrix,
                    particle_counts,

                    training_generation,
                    network_ranking,
                    target_angle,
                    ..
                }) => {
                    if let Some(particle_positions) = particle_positions {
                        self.particle_positions = particle_positions;
                    }
                    if let Some(computation_time) = computation_time {
                        self.computation_time_graph
                            .push_front(computation_time.as_secs_f32() * 1000.);
                        if self.computation_time_graph.len() > 200 {
                            self.computation_time_graph.truncate(200);
                        }
                    }

                    if let Some(power_matrix) = power_matrix {
                        self.power_matrix = power_matrix;
                    }
                    if let Some(particle_counts) = particle_counts {
                        self.particle_counts = particle_counts;
                    }

                    if let Some(gen) = training_generation {
                        self.generation = gen;
                    }
                    if let Some(ranking) = network_ranking {
                        self.network_ranking = Some(ranking);
                    }
                    if let Some(target_angle) = target_angle {
                        self.target_angle = target_angle;
                    }
                }

                Event::TrainingStopped => {
                    self.training_state = TrainingState::Stopped;
                }

                _ => {}
            }
        }

        if self.show_ui {
            SidePanel::left("settings").show(ctx, |ui| {
                ui.heading("settings");
                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .button("respawn")
                        .on_hover_text("spawn particles again")
                        .clicked()
                    {
                        self.senders.send(Recipient::Sim, Event::SpawnParticles);
                    }

                    if self.simulation_state == SimulationState::Running {
                        if ui
                            .button("pause")
                            .on_hover_text("pause the simulation")
                            .clicked()
                        {
                            self.pause();
                        }
                    } else if ui
                        .button("play")
                        .on_hover_text("start the simulation")
                        .clicked()
                    {
                        self.play();
                    }

                    if ui
                        .button("randomize")
                        .on_hover_text("randomly pick a new seed")
                        .clicked()
                    {
                        let w1 = rand::random::<usize>() % self.words.len();
                        let w2 = rand::random::<usize>() % self.words.len();
                        let w3 = rand::random::<usize>() % self.words.len();
                        self.seed =
                            format!("{}_{}_{}", self.words[w1], self.words[w2], self.words[w3]);

                        self.update_history();

                        self.apply_seed();
                        self.senders.send(
                            Recipient::Sim,
                            Event::StateUpdate(
                                StateUpdate::new()
                                    .power_matrix(&self.power_matrix)
                                    .particle_counts(&self.particle_counts),
                            ),
                        );
                        self.senders.send(Recipient::Sim, Event::SpawnParticles);
                    }

                    if ui
                        .button("reset view")
                        .on_hover_text("reset zoom and view position")
                        .clicked()
                    {
                        self.view = View::DEFAULT;
                    }

                    if ui
                        .button("reset")
                        .on_hover_text("reset everything")
                        .clicked()
                    {
                        self.reset();
                    }

                    if ui.button("hide ui").clicked() {
                        self.show_ui = false;
                    }

                    if ui.button("quit").on_hover_text("exit smarticles").clicked() {
                        self.senders.send(Recipient::SimAndTraining, Event::Exit);
                        if let Some(handle) = self.simulation_handle.take() {
                            handle.join().unwrap();
                        };
                        if let Some(handle) = self.training_handle.take() {
                            handle.join().unwrap();
                        }
                        ui.ctx()
                            .send_viewport_cmd(eframe::egui::ViewportCommand::Close);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("seed:");
                    ui.text_edit_singleline(&mut self.seed);
                    if ui.button("apply").clicked() {
                        self.update_history();

                        self.apply_seed();
                        self.senders.send(
                            Recipient::Sim,
                            Event::StateUpdate(
                                StateUpdate::new()
                                    .power_matrix(&self.power_matrix)
                                    .particle_counts(&self.particle_counts),
                            ),
                        );
                        self.senders.send(Recipient::Sim, Event::SpawnParticles);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("total particle count:");

                    let total_particle_count: usize = self.particle_counts.iter().sum();
                    ui.code(total_particle_count.to_string());

                    ui.checkbox(&mut self.locked_particle_counts, "lock particle counts");
                });

                ui.horizontal(|ui| {
                    ui.label("computation time:");
                    if let Some(dt) = self.computation_time_graph.front() {
                        ui.code(format!("{:.1}ms", dt));
                    }
                });

                ui.horizontal(|ui| {
                    Plot::new("computation time")
                        .show_axes(false)
                        .y_axis_label("computation time")
                        .show_x(false)
                        .height(60.)
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_boxed_zoom(false)
                        .allow_scroll(false)
                        .allow_double_click_reset(false)
                        .label_formatter(|_, value| format!("{:.1}ms", value.y))
                        .show(ui, |ui| {
                            ui.line(Line::new(PlotPoints::from_iter(
                                self.computation_time_graph
                                    .iter()
                                    .rev()
                                    .enumerate()
                                    .map(|(i, v)| [i as f64, *v as f64]),
                            )));
                        });
                });

                if self.history.len() > 1 {
                    ui.collapsing("seed history", |ui| {
                        if ComboBox::from_id_salt("seed history")
                            .width(200.)
                            .show_index(
                                ui,
                                &mut self.selected_history_entry,
                                self.history.len(),
                                |i| self.history[i].to_owned(),
                            )
                            .changed()
                        {
                            self.history[self.selected_history_entry].clone_into(&mut self.seed);
                            self.apply_seed();
                            self.senders.send(
                                Recipient::Sim,
                                Event::StateUpdate(
                                    StateUpdate::new()
                                        .power_matrix(&self.power_matrix)
                                        .particle_counts(&self.particle_counts),
                                ),
                            );
                            self.senders.send(Recipient::Sim, Event::SpawnParticles);
                        };
                    });
                }

                ui.collapsing("particle inspector", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("class:");
                        ComboBox::from_id_salt("class").show_index(
                            ui,
                            &mut self.selected_particle.0,
                            self.classes.len(),
                            |i| self.classes[i].heading.to_owned(),
                        );
                        ui.label("particle index:");
                        ui.add(Slider::new(
                            &mut self.selected_particle.1,
                            0..=(self.particle_counts[self.selected_particle.0] - 1),
                        ));
                    });

                    ui.horizontal(|ui| {
                        ui.label("position:");
                        ui.code(format!(
                            "{:?}",
                            self.particle_positions[self.selected_particle]
                        ));
                    });

                    ui.horizontal(|ui| {
                        if self.follow_selected_particle {
                            if ui.button("stop following selected particle").clicked() {
                                self.view.pos -= self.particle_positions[self.selected_particle];
                                self.follow_selected_particle = false;
                            }
                        } else if ui.button("focus and follow selected particle").clicked() {
                            self.view.pos *= 0.;
                            self.follow_selected_particle = true;
                        }
                    });
                });

                ui.collapsing("training menu", |ui| {
                    ui.horizontal(|ui| match self.training_state {
                        TrainingState::Training { target_generation } => {
                            ui.label("generation");
                            ui.code(format!("{} -> {}", self.generation, target_generation));
                            ui.label("training...");
                            ui.spinner();
                        }
                        TrainingState::Evaluating => {
                            ui.label("evaluating networks...");
                            ui.spinner();
                        }
                        _ => {
                            ui.label("generation:");
                            ui.code(format!("{}", self.generation));
                        }
                    });

                    ui.add_enabled_ui(!self.training_state.is_running(), |ui| {
                        if ui.button("evaluate networks").clicked() {
                            self.training_state = TrainingState::Evaluating;
                            self.senders
                                .send(Recipient::Training, Event::EvaluateNetworks);
                        }

                        [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
                            .iter()
                            .copied()
                            .for_each(|gen_count| {
                                if ui
                                    .button(format!("start {} generations", gen_count))
                                    .clicked()
                                {
                                    self.training_state = TrainingState::Training {
                                        target_generation: self.generation + gen_count,
                                    };
                                    self.senders
                                        .send(Recipient::Sim, Event::StartTraining(gen_count));
                                    self.senders
                                        .send(Recipient::Training, Event::StartTraining(gen_count));
                                }
                            });
                    });
                });

                if let Some(ranking) = &self.network_ranking {
                    ui.add_enabled_ui(!self.training_state.is_running(), |ui| {
                        ui.collapsing("inference menu", |ui| {
                            ui.label("network:");
                            ComboBox::from_id_salt("network").width(160.).show_index(
                                ui,
                                &mut self.selected_network,
                                ranking.len(),
                                |i| format!("score: {:.2}", ranking[i].0),
                            );

                            if self.network_state == NetworkState::Running {
                                if ui.button("pause network inference").clicked() {
                                    self.network_state = NetworkState::Stopped;
                                    self.senders.send(Recipient::Sim, Event::NetworkStop);
                                }
                            } else if ui.button("start network inference").clicked() {
                                self.network_state = NetworkState::Running;

                                self.senders.send(
                                    Recipient::Sim,
                                    Event::StateUpdate(
                                        StateUpdate::new()
                                            .inference_network(&ranking[self.selected_network].1),
                                    ),
                                );

                                self.senders.send(Recipient::Sim, Event::NetworkStart);
                            }

                            ui.label("target direction:");

                            ui.vertical_centered(|ui| {
                                if ui.add(DirectionKnob::new(&mut self.target_angle)).changed() {
                                    self.senders.send(
                                        Recipient::Sim,
                                        Event::StateUpdate(
                                            StateUpdate::new().target_angle(self.target_angle),
                                        ),
                                    );
                                };
                            })
                        });
                    });
                }

                ScrollArea::vertical().show(ui, |ui| {
                    for i in 0..CLASS_COUNT {
                        ui.add_space(10.);
                        ui.horizontal(|ui| {
                            ui.colored_label(self.classes[i].color, &self.classes[i].heading);

                            if self.classes[i].enabled {
                                if ui.button("disable").clicked() {
                                    self.classes[i].enabled = false;
                                    self.senders.send(Recipient::Sim, Event::DisableClass(i));
                                }
                            } else if ui.button("enable").clicked() {
                                self.classes[i].enabled = true;
                                self.senders.send(Recipient::Sim, Event::EnsableClass(i));
                            }
                        });
                        ui.separator();

                        if self.classes[i].enabled {
                            ui.horizontal(|ui| {
                                ui.label("particle count:");
                                if ui
                                    .add(Slider::new(
                                        &mut self.particle_counts[i],
                                        MIN_PARTICLE_COUNT..=MAX_PARTICLE_COUNT,
                                    ))
                                    .changed()
                                {
                                    self.senders.send(
                                        Recipient::App,
                                        Event::StateUpdate(
                                            StateUpdate::new()
                                                .particle_counts(&self.particle_counts),
                                        ),
                                    );
                                    self.senders.send(Recipient::Sim, Event::SpawnParticles);
                                }
                            });

                            ui.collapsing(self.classes[i].heading.to_owned() + " params", |ui| {
                                ui.vertical(|ui| {
                                    for j in 0..CLASS_COUNT {
                                        if self.classes[j].enabled {
                                            ui.horizontal(|ui| {
                                                ui.label("power of the force applied on");
                                                let class_name = ui.colored_label(
                                                    self.classes[j].color,
                                                    &self.classes[j].name,
                                                );
                                                ui.add_space(5. - class_name.rect.width() / 2.);
                                                ui.label(":");
                                                ui.add_space(5. - class_name.rect.width() / 2.);
                                                if ui
                                                    .add(Slider::new(
                                                        &mut self.power_matrix[(i, j)],
                                                        MIN_POWER..=MAX_POWER,
                                                    ))
                                                    .changed()
                                                {
                                                    self.seed = self.export_custom_seed();
                                                    self.senders.send(
                                                        Recipient::App,
                                                        Event::StateUpdate(
                                                            StateUpdate::new()
                                                                .power_matrix(&self.power_matrix),
                                                        ),
                                                    );
                                                }
                                                if ui.button("reset").clicked() {
                                                    self.power_matrix[(i, j)] = 0;
                                                    self.senders.send(
                                                        Recipient::App,
                                                        Event::StateUpdate(
                                                            StateUpdate::new()
                                                                .power_matrix(&self.power_matrix),
                                                        ),
                                                    );
                                                }
                                            });
                                        }
                                    }
                                });
                            });
                        }
                    }
                });
            });
        } else {
            Area::new("show_ui_button_area".into())
                .anchor(Align2::LEFT_TOP, [10., 10.]) // Center the button
                .show(ctx, |ui| {
                    if ui.button("show ui").clicked() {
                        self.show_ui = true;
                    }
                });
        }

        CentralPanel::default()
            .frame(eframe::egui::Frame {
                fill: Color32::from_rgba_unmultiplied(12, 10, 10, 255),
                ..Default::default()
            })
            .show(ctx, |ui| {
                let (resp, paint) =
                    ui.allocate_painter(ui.available_size_before_wrap(), Sense::hover());

                if resp
                    .rect
                    .contains(ctx.input(|i| i.pointer.interact_pos()).unwrap_or_default())
                {
                    let scroll_delta = ctx.input(|i| i.smooth_scroll_delta).y;
                    if scroll_delta > 0. {
                        self.view.zoom *= ZOOM_FACTOR;
                    } else if scroll_delta < 0. {
                        self.view.zoom /= ZOOM_FACTOR;
                    }
                }

                // This is weird but look at the values.
                self.view.zoom = self.view.zoom.clamp(MIN_ZOOM, MAX_ZOOM);

                let center = resp.rect.center().to_vec2()
                    + if self.follow_selected_particle {
                        self.view.pos - self.particle_positions[self.selected_particle]
                    } else {
                        self.view.pos
                    } * self.view.zoom;

                if let Some(interact_pos) = ctx.input(|i| i.pointer.interact_pos()) {
                    if self.view.dragging {
                        let drag_delta = interact_pos - self.view.drag_start_pos;
                        self.view.pos =
                            self.view.drag_start_view_pos + drag_delta.to_vec2() / self.view.zoom;
                    }
                    if ctx.input(|i| i.pointer.button_down(PointerButton::Primary))
                        && resp.rect.contains(interact_pos)
                    {
                        if !self.view.dragging {
                            self.view.dragging = true;
                            self.view.drag_start_pos = interact_pos.to_vec2();
                            self.view.drag_start_view_pos = self.view.pos;
                        }
                    } else {
                        self.view.dragging = false;
                    }
                }

                for c in (0..CLASS_COUNT).filter(|c| self.classes[*c].enabled) {
                    let class = &self.classes[c];

                    for p in 0..self.particle_counts[c] {
                        let pos =
                            (center + self.particle_positions[(c, p)] * self.view.zoom).to_pos2();

                        paint.circle_filled(
                            pos,
                            if (c, p) == self.selected_particle && self.classes[c].enabled {
                                PARTICLE_DIAMETER * 3.
                            } else {
                                PARTICLE_DIAMETER
                            } * self.view.zoom,
                            class.color,
                        );
                    }
                }
            });

        ctx.request_repaint();
    }
}
