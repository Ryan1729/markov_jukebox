extern crate find_folder;
extern crate hound;
extern crate portaudio as pa;
extern crate sample;
extern crate rand;

use sample::{signal, Signal, ToFrameSliceMut};

use rand::{Rng, SeedableRng, StdRng};


pub const NUM_CHANNELS: usize = 2;
pub type Frame = [i16; NUM_CHANNELS];

const FRAMES_PER_BUFFER: u32 = 64;
const SAMPLE_RATE: f64 = 44_100.0;


extern crate clap;
use clap::{App, Arg};

fn main() {
    let matches = App::new("markov_jukebox")
        .arg(Arg::with_name("filenames").takes_value(true).required(true))
        .arg(
            Arg::with_name("play")
                .short("p")
                .help("play the files before processing them"),
        )
        .get_matches();

    let play = matches.is_present("play");

    let seed: &[_] = &[42];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    if let Some(filenames) = matches.values_of("filenames") {
        run(filenames.collect(), play, &mut rng).unwrap();
    } else {
        println!("No filenames recieved");
    }

}

fn run<R: Rng>(filenames: Vec<&str>, play: bool, rng: &mut R) -> Result<(), pa::Error> {
    if play {
        // Initialise PortAudio.
        let pa = try!(pa::PortAudio::new());
        let settings = try!(pa.default_output_stream_settings::<i16>(
            NUM_CHANNELS as i32,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
        ));

        for filename in filenames.iter() {
            // Get the frames to play back.
            let frames: Vec<Frame> = read_frames(filename);
            let mut signal = frames.clone().into_iter();

            // Define the callback which provides PortAudio the audio.
            let callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
                let buffer: &mut [Frame] = buffer.to_frame_slice_mut().unwrap();
                for out_frame in buffer {
                    match signal.next() {
                        Some(frame) => *out_frame = frame,
                        None => return pa::Complete,
                    }
                }
                pa::Continue
            };

            let mut stream = try!(pa.open_non_blocking_stream(settings, callback));
            try!(stream.start());

            while let Ok(true) = stream.is_active() {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            try!(stream.stop());
            try!(stream.close());

            let blended_frames = blend_frames(&frames, rng);
            write_frames(&blended_frames, None);
        }
    } else {
        for filename in filenames.iter() {
            let frames: Vec<Frame> = read_frames(filename);
            let blended_frames = blend_frames(&frames, rng);
            write_frames(&blended_frames, None);
        }
    }


    Ok(())
}

// Given the file name, produces a Vec of `Frame`s which may be played back.
fn read_frames(file_name: &str) -> Vec<Frame> {
    println!("Loading {}", file_name);

    let mut reader = hound::WavReader::open(file_name).unwrap();
    let spec = reader.spec();
    let duration = reader.duration();
    let new_duration = (duration as f64 * (SAMPLE_RATE as f64 / spec.sample_rate as f64)) as usize;
    let samples = reader.samples().map(|s| s.unwrap());
    let signal = signal::from_interleaved_samples::<_, Frame>(samples);
    signal
        .from_hz_to_hz(spec.sample_rate as f64, SAMPLE_RATE as f64)
        .take(new_duration)
        .collect()
}

const SILENCE: Frame = [0; NUM_CHANNELS];

fn blend_frames<R: Rng>(frames: &Vec<Frame>, rng: &mut R) -> Vec<Frame> {
    let len = frames.len();

    if len == 0 {
        return Vec::new();
    }

    println!("get_next_frames");

    let next_frames = get_next_frames(frames);

    println!("done get_next_frames");

    let mut result = Vec::with_capacity(len);

    let mut previous = (frames[0], frames[1]);
    for i in 0..2 {
        result.push(frames[i]);
    }

    rng.gen_range(0, 12);

    let mut count = 0;
    println!("{} < {} ", count, len);
    while count < len {
        let choices = get_choices(&next_frames, previous);
        println!("choices {:?}", choices);

        // let next = *rng.choose(&choices).unwrap();
        // let next = *choices.last().unwrap();
        let next = *choices
            .iter()
            .max_by(|f1, f2| magnitude(f1).cmp(&magnitude(f2)))
            .unwrap();

        println!("{:?}", next);
        if next == SILENCE {
            println!("{:?} to SILENCE", previous);
        }

        result.push(next);

        previous = (previous.1, next);

        count += 1;
    }
    println!("{}", result.len());
    result
}

const MINIMUM_CHOICES: usize = 5;

// https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
fn get_choices(next_frames: &NextFrames, previous: (Frame, Frame)) -> Vec<Frame> {
    let mut depth = 0;

    let mut result = Vec::new();

    let left = (-1, 0);
    let up = (0, 1);
    let right = (1, 0);
    let down = (0, -1);

    let upward = [left, up, right];
    let downward = [right, down, left];
    //`upward` and `downward` will cover the space that these would cover
    //if they inculded `up` and `down`
    let rightward = [right];
    let leftward = [left];

    while depth <= 16 {
        get_choices_helper(next_frames, previous, &mut result, &leftward, depth);
        get_choices_helper(next_frames, previous, &mut result, &upward, depth);
        get_choices_helper(next_frames, previous, &mut result, &rightward, depth);
        get_choices_helper(next_frames, previous, &mut result, &downward, depth);

        if result.len() >= MINIMUM_CHOICES {
            return result;
        }

        println!("depth {}", depth);
        depth += 1;
    }

    result
}

fn get_choices_helper(
    next_frames: &NextFrames,
    current: (Frame, Frame),
    result: &mut Vec<Frame>,
    offsets: &[(i16, i16)],
    depth: usize,
) {
    if depth == 0 {
        if let Some(choices) = next_frames.get(&current) {
            result.extend(choices);
            result.dedup();
        }

        return;
    }

    // println!("current {:?} depth {}", current, depth);

    for &(offset_0, offset_1) in offsets.iter() {
        let new_key = (
            saturating_add(current.0, offset_0),
            saturating_add(current.1, offset_1),
        );

        get_choices_helper(next_frames, new_key, result, offsets, depth - 1);

        if result.len() >= MINIMUM_CHOICES {
            return;
        }
    }
}

use std::collections::HashMap;

type NextFrames = HashMap<(Frame, Frame), Vec<Frame>>;

fn get_next_frames(frames: &Vec<Frame>) -> NextFrames {
    let mut result = HashMap::new();

    for window in frames.windows(3) {
        result
            .entry((window[0], window[1]))
            .or_insert(Vec::new())
            .push(window[2]);
    }

    {
        let silence_target = result.entry((SILENCE, SILENCE)).or_insert(Vec::new());

        silence_target.retain(|&frame| frame != SILENCE);

        if silence_target.len() == 0 {
            let middle = frames[frames.len() / 2];

            if middle == SILENCE {
                for &frame in frames.iter() {
                    if magnitude(&frame) > 256 {
                        silence_target.push(frame);
                        break;
                    }
                }
            } else {
                silence_target.push(middle);
            }

            println!("{:?}", silence_target);
        }
    }

    result
}

use std::cmp::max;

fn magnitude(frame: &Frame) -> i16 {

    frame
        .iter()
        .fold(0, |acc, channel_value| max(acc, channel_value.abs()))
}

fn saturating_add(frame: Frame, x: i16) -> Frame {
    [frame[0].saturating_add(x), frame[1].saturating_add(x)]
}
fn saturating_sub(frame: Frame, x: i16) -> Frame {
    [frame[0].saturating_sub(x), frame[1].saturating_sub(x)]
}


fn key_round_frame(frame: Frame) -> Frame {
    [key_round(frame[0]), key_round(frame[1])]
}

fn key_round(x: i16) -> i16 {
    // if x & 0b10 == 0 {
    //     x & !0b11
    // } else {
    //     (x | 0b11).saturating_add(1)
    // }
    if x & 1 == 0 {
        x
    } else {
        (x).saturating_add(1)
    }

}

fn write_frames(frames: &Vec<Frame>, optional_name: Option<&str>) {
    if let Some(name) = optional_name {
        write_frames_with_name(frames, name)
    } else {
        let name = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .to_string();

        write_frames_with_name(frames, &name)
    }
}

fn write_frames_with_name(frames: &Vec<Frame>, name: &str) {
    let mut path = std::path::PathBuf::new();
    path.push("output");
    path.push(name);
    path.set_extension("wav");

    println!("Writing to {:?}", path.to_str().unwrap());

    let spec = hound::WavSpec {
        channels: NUM_CHANNELS as _,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).unwrap();

    for frame in frames.iter() {
        for channel in 0..NUM_CHANNELS {
            writer.write_sample(frame[channel]).unwrap();
        }
    }

    writer.finalize().unwrap();
}
