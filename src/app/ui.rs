use std::{
    collections::{hash_map::DefaultHasher, VecDeque},
    hash::{Hash, Hasher},
    sync::mpsc::Receiver,
    thread::JoinHandle,
};

use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use eframe::epaint::Color32;
use eframe::{App, Frame};
use egui::{
    plot::{Line, Plot, PlotPoints},
    Button, Vec2,
};
use egui::{
    Align2, CentralPanel, ComboBox, Context, FontId, ScrollArea, Sense, SidePanel, Slider, Stroke,
};
use rand::distributions::Open01;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    ai::net::Network,
    simulation::{calculate_force, FIRST_THRESHOLD, SECOND_THRESHOLD},
};
use crate::{mat::Mat2D, Senders};
use crate::{
    SmarticlesEvent, CLASS_COUNT, MAX_FORCE, MAX_PARTICLE_COUNT, MIN_FORCE, MIN_PARTICLE_COUNT,
    RANDOM_MAX_PARTICLE_COUNT, RANDOM_MIN_PARTICLE_COUNT,
};

use super::{
    simulation::{NetworkState, SimulationState},
    training::random_target_position,
};

/// Display diameter of the particles in the simulation (in
/// pixels).
const PARTICLE_DIAMETER: f32 = 0.5;

const DEFAULT_ZOOM: f32 = 2.;
const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 30.;
const ZOOM_FACTOR: f32 = 1.08;

const MAX_HISTORY_LEN: usize = 10;

#[derive(Debug, PartialEq)]
enum TrainingState {
    Running { target_generation: usize },
    Stopped,
}
impl TrainingState {
    pub fn is_running(&self) -> bool {
        match self {
            TrainingState::Running { .. } => true,
            _ => false,
        }
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
}

pub struct Ui {
    classes: [ClassProps; CLASS_COUNT],

    seed: String,

    view: View,

    selected_param: (usize, usize),
    selected_particle: (usize, usize),
    follow_selected_particle: bool,

    history: VecDeque<String>,
    selected_history_entry: usize,

    calculation_time: u128,

    words: Vec<String>,

    senders: Senders,
    receiver: Receiver<SmarticlesEvent>,

    simulation_handle: Option<JoinHandle<()>>,
    training_handle: Option<JoinHandle<()>>,

    // TODO: group the next fields into a new struct
    particle_positions: Mat2D<Vec2>,
    particle_counts: [usize; CLASS_COUNT],
    force_matrix: Mat2D<f32>,
    simulation_state: SimulationState,
    network_state: NetworkState,
    network_ranking: Option<Vec<(f32, Network)>>,
    selected_network: usize,
    target_position: Vec2,
    training_state: TrainingState,
    generation: usize,
}

impl Ui {
    pub fn new<S>(
        classes: [(S, Color32); CLASS_COUNT],
        generation: usize,
        senders: Senders,
        receiver: Receiver<SmarticlesEvent>,
        simulation_handle: Option<JoinHandle<()>>,
        training_handle: Option<JoinHandle<()>>,
    ) -> Self
    where
        S: ToString,
    {
        let words = include_str!("words.txt");
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

        let target_position = random_target_position(Vec2::ZERO);
        senders.send_sim(SmarticlesEvent::TargetPositionChange(target_position));

        Self {
            seed: "".to_string(),

            classes: classes.map(|(name, color)| ClassProps {
                name: name.to_string(),
                heading: "class ".to_string() + &name.to_string(),
                color,
            }),

            view: View::DEFAULT,

            selected_param: (0, 0),
            selected_particle: (0, 0),
            follow_selected_particle: false,

            history: VecDeque::new(),
            selected_history_entry: 0,

            calculation_time: 0,

            words,

            senders,
            receiver,

            simulation_handle,
            training_handle,

            particle_positions: Mat2D::filled_with(Vec2::ZERO, CLASS_COUNT, MAX_PARTICLE_COUNT),
            particle_counts: [200; CLASS_COUNT],
            force_matrix: Mat2D::filled_with(0., CLASS_COUNT, CLASS_COUNT),
            simulation_state: SimulationState::Paused,
            network_state: NetworkState::Stopped,
            network_ranking: None,
            selected_network: 0,
            target_position,
            training_state: TrainingState::Stopped,
            generation,
        }
    }

    fn apply_seed(&mut self) {
        let mut rand = if self.seed.is_empty() {
            SmallRng::from_entropy()
        } else {
            if self.seed.starts_with('@') {
                if let Ok(bytes) = base64::decode(&self.seed[1..]) {
                    self.import(&bytes);
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
            self.particle_counts[i] = rand(
                RANDOM_MIN_PARTICLE_COUNT as f32,
                RANDOM_MAX_PARTICLE_COUNT as f32,
            ) as usize;
            for j in 0..CLASS_COUNT {
                let pow = rand(MIN_FORCE, MAX_FORCE);
                self.force_matrix[(i, j)] = pow.signum() * pow.abs().powf(1. / POW_F);
            }
        }

        self.send_params();
        self.send_particle_counts();
    }

    fn send_params(&self) {
        self.senders.send_sim(SmarticlesEvent::ForceMatrixChange(
            self.force_matrix.to_owned(),
        ));
    }
    fn send_particle_counts(&self) {
        self.senders.send_sim(SmarticlesEvent::ParticleCountsUpdate(
            self.particle_counts.to_owned(),
        ));
    }

    fn export(&self) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for count in &self.particle_counts {
            bytes.write_u16::<LE>(*count as u16).unwrap();
        }
        self.force_matrix.vec().iter().copied().for_each(|force| {
            bytes.write_i8(force as i8).unwrap();
        });

        format!("@{}", base64::encode(bytes))
    }

    fn import(&mut self, mut bytes: &[u8]) {
        for count in &mut self.particle_counts {
            *count = bytes.read_u16::<LE>().unwrap_or(0) as usize;
        }

        for i in 0..CLASS_COUNT {
            for j in 0..CLASS_COUNT {
                self.force_matrix[(i, j)] = bytes.read_i8().unwrap_or(0) as f32;
            }
        }
    }

    fn update_history(&mut self) {
        self.history.push_front(self.seed.to_owned());
        if self.history.len() > MAX_HISTORY_LEN {
            self.history.pop_back();
        }
        self.selected_history_entry = 0;
    }

    fn play(&mut self) {
        self.simulation_state = SimulationState::Running;
        self.senders.send_sim(SmarticlesEvent::SimulationStart);
    }
    fn pause(&mut self) {
        self.simulation_state = SimulationState::Paused;
        self.senders.send_sim(SmarticlesEvent::SimulationPause);
    }
    fn reset(&mut self) {
        self.simulation_state = SimulationState::Paused;
        self.senders.send_sim(SmarticlesEvent::SimulationReset);
    }
    fn spawn(&mut self) {
        self.senders.send_sim(SmarticlesEvent::SpawnParticles);
    }
}

impl App for Ui {
    fn update(&mut self, ctx: &Context, frame: &mut Frame) {
        let events = self.receiver.try_iter();
        for event in events {
            match event {
                SmarticlesEvent::SimulationResults(positions, elapsed) => {
                    if let Some(elapsed) = elapsed {
                        self.calculation_time = elapsed.as_millis();
                    }
                    self.particle_positions = positions;
                }

                SmarticlesEvent::ForceMatrixChange(force_matrix) => {
                    self.force_matrix = force_matrix;
                }

                SmarticlesEvent::ParticleCountsUpdate(particle_counts) => {
                    self.particle_counts = particle_counts;
                }

                SmarticlesEvent::GenerationChange(generation) => {
                    self.generation = generation;
                }
                SmarticlesEvent::NetworkRanking(networks_and_scores) => {
                    self.training_state = TrainingState::Stopped;
                    self.network_ranking = Some(networks_and_scores);
                }

                _ => {}
            }
        }

        SidePanel::left("settings").show(ctx, |ui| {
            ui.heading("settings");
            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .button("respawn")
                    .on_hover_text("spawn particles again")
                    .clicked()
                {
                    self.spawn();
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
                    self.seed = format!("{}_{}_{}", self.words[w1], self.words[w2], self.words[w3]);

                    self.update_history();

                    self.apply_seed();
                    self.spawn();
                }

                if ui
                    .button("reset View")
                    .on_hover_text("reset zoom and position")
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

                if ui.button("quit").on_hover_text("exit smarticles").clicked() {
                    self.senders.send_sim(SmarticlesEvent::Quit);
                    self.senders.send_training(SmarticlesEvent::Quit);
                    if let Some(handle) = self.simulation_handle.take() {
                        handle.join().unwrap();
                    }
                    if let Some(handle) = self.training_handle.take() {
                        handle.join().unwrap();
                    }
                    frame.close();
                }
            });
            ui.horizontal(|ui| {
                ui.label("seed:");
                ui.text_edit_singleline(&mut self.seed);
                if ui.button("apply").clicked() {
                    self.update_history();

                    self.apply_seed();
                    self.spawn();
                }
            });

            ui.horizontal(|ui| {
                ui.label("total particle count:");

                let total_particle_count: usize = self.particle_counts.iter().sum();
                ui.code(total_particle_count.to_string());
            });

            ui.horizontal(|ui| {
                ui.label("calculation time:");
                ui.code(self.calculation_time.to_string() + "ms");
            });

            if self.history.len() > 1 {
                ui.collapsing("seed history", |ui| {
                    if ComboBox::from_id_source("seed history")
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
                        self.spawn();
                    };
                });
            }

            ui.collapsing("particle inspector", |ui| {
                ui.horizontal(|ui| {
                    ui.label("class:");
                    ComboBox::from_id_source("class").show_index(
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

            ui.collapsing(
                "velocity elementary variation with respect to distance between particles",
                |ui| {
                    let points: PlotPoints = (0
                        ..((FIRST_THRESHOLD + 2. * SECOND_THRESHOLD) * 1000.) as i32)
                        .map(|i| {
                            let x = i as f32 / 1000.;
                            [
                                x as f64,
                                calculate_force(x, self.force_matrix[self.selected_param]) as f64,
                            ]
                        })
                        .collect();
                    let line = Line::new(points);
                    Plot::new("activation function")
                        .view_aspect(2.0)
                        .show(ui, |plot_ui| plot_ui.line(line));
                },
            );

            ui.collapsing("training menu", |ui| {
                ui.horizontal(|ui| {
                    if let TrainingState::Running { target_generation } = self.training_state {
                        ui.label("generation");
                        ui.code(format!("{} -> {}", self.generation, target_generation));
                        ui.label("training...");
                        ui.spinner();
                    } else {
                        ui.label("generation:");
                        ui.code(format!("{}", self.generation));
                    }
                });

                if ui
                    .add_enabled(
                        !self.training_state.is_running(),
                        Button::new("evaluate networks"),
                    )
                    .clicked()
                {
                    self.senders
                        .send_training(SmarticlesEvent::EvaluateNetworks);
                }

                [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
                    .iter()
                    .copied()
                    .for_each(|gen_count| {
                        if ui
                            .add_enabled(
                                !self.training_state.is_running(),
                                Button::new(format!("start {} generations", gen_count)),
                            )
                            .clicked()
                        {
                            self.training_state = TrainingState::Running {
                                target_generation: self.generation + gen_count,
                            };
                            self.senders
                                .send_sim(SmarticlesEvent::StartTraining(gen_count));
                            self.senders
                                .send_training(SmarticlesEvent::StartTraining(gen_count));
                        }
                    });

                if let Some(ranking) = &self.network_ranking {
                    ui.label("network:");
                    ComboBox::from_id_source("network").width(160.).show_index(
                        ui,
                        &mut self.selected_network,
                        ranking.len(),
                        |i| format!("score: {:.0}", ranking[i].0),
                    );

                    if self.network_state == NetworkState::Running {
                        if ui
                            .add_enabled(
                                !self.training_state.is_running(),
                                Button::new("pause network inference"),
                            )
                            .clicked()
                        {
                            self.network_state = NetworkState::Stopped;
                            self.senders.send_sim(SmarticlesEvent::NetworkStop);
                        }
                    } else if ui
                        .add_enabled(
                            !self.training_state.is_running(),
                            Button::new("start network inference"),
                        )
                        .clicked()
                    {
                        self.network_state = NetworkState::Running;
                        self.senders
                            .send_sim(SmarticlesEvent::InferenceNetworkChange(
                                ranking[self.selected_network].1.clone(),
                            ));
                        self.senders.send_sim(SmarticlesEvent::NetworkStart);
                    }
                }
            });

            ScrollArea::vertical().show(ui, |ui| {
                for i in 0..CLASS_COUNT {
                    ui.add_space(10.);
                    ui.colored_label(self.classes[i].color, &self.classes[i].heading);
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("particle count:");
                        if ui
                            .add(Slider::new(
                                &mut self.particle_counts[i],
                                MIN_PARTICLE_COUNT..=MAX_PARTICLE_COUNT,
                            ))
                            .changed()
                        {
                            self.seed = self.export();
                            self.spawn();

                            self.send_particle_counts();
                        }
                    });

                    ui.collapsing(self.classes[i].heading.to_owned() + " params", |ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                for j in 0..CLASS_COUNT {
                                    ui.horizontal(|ui| {
                                        ui.label("force (");
                                        ui.colored_label(
                                            self.classes[j].color,
                                            &self.classes[j].name,
                                        );
                                        ui.label(")");
                                        if ui
                                            .add(Slider::new(
                                                &mut self.force_matrix[(i, j)],
                                                MIN_FORCE..=MAX_FORCE,
                                            ))
                                            .changed()
                                        {
                                            self.selected_param = (i, j);
                                            self.seed = self.export();

                                            self.send_params();
                                        }
                                    });
                                }
                            });
                        });
                    });
                }
            });
        });

        CentralPanel::default()
            .frame(egui::Frame {
                fill: Color32::from_rgba_unmultiplied(12, 12, 12, 255),
                ..Default::default()
            })
            .show(ctx, |ui| {
                let (resp, paint) =
                    ui.allocate_painter(ui.available_size_before_wrap(), Sense::hover());

                if resp
                    .rect
                    .contains(ctx.input().pointer.interact_pos().unwrap_or_default())
                {
                    if ctx.input().scroll_delta.y > 0.0 {
                        self.view.zoom *= ZOOM_FACTOR;
                    } else if ctx.input().scroll_delta.y < 0.0 {
                        self.view.zoom /= ZOOM_FACTOR;
                    }
                }

                // This is weird but look at the values.
                self.view.zoom = self.view.zoom.clamp(MIN_ZOOM, MAX_ZOOM);

                if self.view.dragging {
                    let drag_delta =
                        ctx.input().pointer.interact_pos().unwrap() - self.view.drag_start_pos;
                    self.view.pos =
                        self.view.drag_start_view_pos + drag_delta.to_vec2() / self.view.zoom;
                }

                let center = resp.rect.center().to_vec2()
                    + if self.follow_selected_particle {
                        self.view.pos - self.particle_positions[self.selected_particle]
                    } else {
                        self.view.pos
                    } * self.view.zoom;

                if let Some(interact_pos) = ctx.input().pointer.interact_pos() {
                    if ctx
                        .input()
                        .pointer
                        .button_down(egui::PointerButton::Primary)
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

                    if ctx
                        .input()
                        .pointer
                        .button_clicked(egui::PointerButton::Secondary)
                    {
                        self.target_position = (interact_pos.to_vec2() - center) / self.view.zoom;
                        self.senders
                            .send_sim(SmarticlesEvent::TargetPositionChange(self.target_position));
                    }
                }

                paint.circle_stroke(
                    (center + self.target_position * self.view.zoom).to_pos2(),
                    25. * self.view.zoom,
                    Stroke::new(2., Color32::WHITE),
                );

                let pos = Vec2::new(resp.rect.right() - 30., resp.rect.bottom() - 10.);
                paint.line_segment(
                    [
                        (pos - Vec2::new(10. * self.view.zoom, 0.)).to_pos2(),
                        pos.to_pos2(),
                    ],
                    Stroke::new(4., Color32::WHITE),
                );
                paint.text(
                    (pos + Vec2::new(10., 0.)).to_pos2(),
                    Align2::LEFT_CENTER,
                    "10",
                    FontId::monospace(14.),
                    Color32::WHITE,
                );

                for c in 0..CLASS_COUNT {
                    let class = &self.classes[c];

                    for p in 0..self.particle_counts[c] {
                        let pos =
                            (center + self.particle_positions[(c, p)] * self.view.zoom).to_pos2();
                        if paint.clip_rect().contains(pos) {
                            paint.circle_filled(
                                pos,
                                if (c, p) == self.selected_particle {
                                    PARTICLE_DIAMETER * 3.
                                } else {
                                    PARTICLE_DIAMETER
                                } * self.view.zoom,
                                class.color,
                            );
                        }
                    }
                }

                // if self.shared.simulation_state != SimulationState::Stopped {
                //     paint.circle_stroke(
                //         center
                //             + self.shared.particle_positions[self.selected_particle]
                //                 * self.view.zoom,
                //         PARTICLE_DIAMETER + 4.,
                //         Stroke::new(1., Color32::WHITE),
                //     );
                // }
            });

        ctx.request_repaint();
    }
}
