import os
import pretty_midi as pm
import numpy as np
import csv

def extract_absolute_pitch_and_rhythm(midi_file):
    try:
        midi_data = pm.PrettyMIDI(midi_file)
        
        absolute_pitch_sequence = []
        absolute_rhythm_sequence = []

        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.velocity > 0:  # Filter out silences
                        absolute_pitch_sequence.append(note.pitch)
                        absolute_rhythm_sequence.append(round(note.end - note.start, 2))

        if len(absolute_pitch_sequence) == 0 or len(absolute_rhythm_sequence) == 0:
            raise ValueError("No valid pitch or rhythm data found.")

        return np.array(absolute_pitch_sequence), np.array(absolute_rhythm_sequence)
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None, None

def compute_relative_pitch(pitch_sequence):
    return np.diff(pitch_sequence)

def compute_relative_rhythm(rhythm_sequence):
    relative_rhythm = np.array([rhythm_sequence[i+1] / rhythm_sequence[i] for i in range(len(rhythm_sequence) - 1)])
    return np.around(relative_rhythm, decimals=2)

def sliding_window(sequence, window_size=8, step_size=4):
    """
    Splits the given sequence into overlapping sub-sequences using a sliding window approach.
    
    Args:
        sequence (np.ndarray): The input sequence (either relative pitch or rhythm).
        window_size (int): The size of each window (default: 8).
        step_size (int): The number of elements to shift the window by (default: 4).
        
    Returns:
        np.ndarray: Array of sliding window sequences.
    """
    windows = [
        sequence[i : i + window_size]
        for i in range(0, len(sequence) - window_size + 1, step_size)
    ]
    return np.array(windows)

def get_file_info(file_path, base_folder):
    """
    Extracts detailed information about the MIDI file based on its path.
    
    Args:
        file_path (str): Full path to the MIDI file
        base_folder (str): Base folder (Plagiarism or No_Plagiarism)
        
    Returns:
        dict: Dictionary containing file information
    """
    rel_path = os.path.relpath(os.path.dirname(file_path), base_folder)
    folders = rel_path.split(os.sep)
    
    # Extract case number if present in folder name
    case_info = next((folder for folder in folders if "case" in folder.lower()), "Unknown Case")
    
    # Determine ruling based on the base folder path
    ruling = "Plagiarism" if "Plagiarism" in base_folder and "No_Plagiarism" not in base_folder else "No_Plagiarism"
    
    return {
        "ruling": ruling,
        "case": case_info,
        "subfolder_path": rel_path,
        "filename": os.path.basename(file_path)
    }

def store_library(library_path, data):
    """Save library with separate n-gram counts for pitch and rhythm."""
    with open(library_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "File Name", "Ruling", "Case",
            "Absolute Pitch", "Absolute Rhythm", 
            "Relative Pitch", "Relative Rhythm",
            "Pitch N-grams", "Rhythm N-grams"  # Changed column names
        ])
        for file_name, info in data.items():
            # Get separate counts for pitch and rhythm n-grams
            pitch_grams = len(info["relative_pitch"]) if isinstance(info["relative_pitch"], np.ndarray) else 0
            rhythm_grams = len(info["relative_rhythm"]) if isinstance(info["relative_rhythm"], np.ndarray) else 0
            
            writer.writerow([
                file_name,
                info["file_info"]["ruling"],
                info["file_info"]["case"],  
                info["absolute_pitch"].tolist(),
                info["absolute_rhythm"].tolist(),
                info["relative_pitch"].tolist(),
                info["relative_rhythm"].tolist(),
                pitch_grams,  # Separate pitch n-gram count
                rhythm_grams  # Separate rhythm n-gram count
            ])

def store_library_unsegmented(library_path, data):
    """
    Store melody library without n-gram segmentation.
    Each song has one sequence list for each feature.
    """
    with open(library_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "File Name", "Ruling", "Case",
            "Absolute Pitch", "Absolute Rhythm", 
            "Relative Pitch", "Relative Rhythm"
        ])
        
        for file_name, info in data.items():
            # Convert numpy arrays to regular lists without data type info
            abs_pitch = info["absolute_pitch"][0].tolist()
            abs_rhythm = info["absolute_rhythm"][0].tolist()
            
            # Get first absolute sequence and compute relative sequences
            rel_pitch = compute_relative_pitch(info["absolute_pitch"][0]).tolist()
            rel_rhythm = compute_relative_rhythm(info["absolute_rhythm"][0]).tolist()
            
            writer.writerow([
                file_name,
                info["file_info"]["ruling"],
                info["file_info"]["case"],
                str(abs_pitch),
                str(abs_rhythm),
                str(rel_pitch),
                str(rel_rhythm)
            ])

def save_subarrays(data, output_dir, file_name, chunk_size=1):
    """
    Saves the given data into separate .npy files as sub-arrays.
    """
    save_folder = os.path.join(output_dir, file_name)
    os.makedirs(save_folder, exist_ok=True)

    for idx, sub_array in enumerate(data):
        save_path = os.path.join(save_folder, f"{file_name}_gram_{idx}.npy")
        np.save(save_path, sub_array)

def get_sequence_parameters():
    """Get user input for sliding window parameters."""
    # Get window size
    while True:
        print("\nSelect window size for n-gram sequences:")
        print("1. Window size = 2 (Minimum)")
        print("2. Window size = 3")
        print("3. Window size = 4")
        print("4. Window size = 5")
        print("5. Window size = 6")
        print("6. Window size = 7")
        print("7. Window size = 8 (Maximum)")
        
        try:
            choice = int(input("\nEnter your choice (1-7): "))
            if 1 <= choice <= 7:
                window_size = choice + 1  # Maps choice 1-7 to sizes 2-8
                break
            print("Invalid choice! Please select 1-7.")
        except ValueError:
            print("Invalid input! Please enter a number between 1-7.")
    
    # Get step size
    while True:
        print("\nSelect step size for sliding window:")
        print("1. Step size = 1 (Maximum overlap)")
        print("2. Step size = 2")
        print("3. Step size = 3")
        print("4. Step size = 4 (Default)")
        print("5. Step size = 5 (Minimum overlap)")
        
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            if 1 <= choice <= 5:
                step_size = choice
                break
            print("Invalid choice! Please select 1-5.")
        except ValueError:
            print("Invalid input! Please enter a number between 1-5.")
    
    return window_size, step_size

def process_file(midi_path, abs_output_dir, rel_output_dir, file_name, library, file_info, window_size, step_size):
    pitch_file = os.path.join(abs_output_dir, f"{file_name}_pitch.npy")
    rhythm_file = os.path.join(abs_output_dir, f"{file_name}_rhythm.npy")
    rel_pitch_file = os.path.join(rel_output_dir, f"{file_name}_relative_pitch.npy")
    rel_rhythm_file = os.path.join(rel_output_dir, f"{file_name}_relative_rhythm.npy")
    
    # Load or extract sequences
    if os.path.exists(pitch_file) and os.path.exists(rhythm_file):
        try:
            pitch_sequence = np.load(pitch_file)
            rhythm_sequence = np.load(rhythm_file)
        except Exception as e:
            print(f"Error loading sequences for {file_name}: {e}")
            return
    else:
        pitch_sequence, rhythm_sequence = extract_absolute_pitch_and_rhythm(midi_path)
        
        if pitch_sequence is not None and rhythm_sequence is not None:
            # Apply sliding window to original sequences
            try:
                # Apply sliding window with user-defined parameters
                pitch_sequence = sliding_window(pitch_sequence, window_size=window_size, step_size=step_size)
                rhythm_sequence = sliding_window(rhythm_sequence, window_size=window_size, step_size=step_size)
                
                if len(pitch_sequence) == 0 or len(rhythm_sequence) == 0:
                    return
                    
                pitch_sequence = np.around(pitch_sequence, decimals=2)
                rhythm_sequence = np.around(rhythm_sequence, decimals=2)
                np.save(pitch_file, pitch_sequence)
                np.save(rhythm_file, rhythm_sequence)
            except Exception as e:
                print(f"Error processing sequences for {file_name}: {e}")
                return
        else:
            return
    
    os.makedirs(rel_output_dir, exist_ok=True)
    
    # Process relative pitch
    if not os.path.exists(os.path.join(rel_output_dir, f"{file_name}_relative_pitch")):
        try:
            # Compute relative pitch for each window
            relative_pitch = np.array([compute_relative_pitch(window) for window in pitch_sequence])
            if len(relative_pitch) > 0:
                relative_pitch = np.around(relative_pitch, decimals=2)
                save_subarrays(relative_pitch, rel_output_dir, f"{file_name}_relative_pitch")
            else:
                return
        except Exception as e:
            print(f"Error computing relative pitch for {file_name}: {e}")
            return
    
    # Process relative rhythm
    if not os.path.exists(os.path.join(rel_output_dir, f"{file_name}_relative_rhythm")):
        try:
            # Compute relative rhythm for each window
            relative_rhythm = np.array([compute_relative_rhythm(window) for window in rhythm_sequence])
            if len(relative_rhythm) > 0:
                relative_rhythm = np.around(relative_rhythm, decimals=2)
                save_subarrays(relative_rhythm, rel_output_dir, f"{file_name}_relative_rhythm")
            else:
                return
        except Exception as e:
            print(f"Error computing relative rhythm for {file_name}: {e}")
            return
    
    try:
        # Count n-grams silently for library storage
        pitch_grams = len(relative_pitch) if isinstance(relative_pitch, np.ndarray) else 0
        rhythm_grams = len(relative_rhythm) if isinstance(relative_rhythm, np.ndarray) else 0
        
        library[file_name] = {
            "file_info": file_info,
            "absolute_pitch": pitch_sequence,
            "absolute_rhythm": rhythm_sequence,
            "relative_pitch": relative_pitch,
            "relative_rhythm": relative_rhythm
        }
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")

def process_folder(input_folder, abs_output_folder, rel_output_folder, library_path):
    """Process folder with user-defined window and step sizes."""
    library = {}
    
    # Get window and step sizes from user
    window_size, step_size = get_sequence_parameters()
    print(f"\nUsing:")
    print(f"- Window size: {window_size}")
    print(f"- Step size: {step_size}")
    print(f"This will affect the number of n-gram sequences generated per song.")
    
    # Process Plagiarism and No_Plagiarism folders
    for ruling in ["Plagiarism", "No_Plagiarism"]:  # Changed from category to ruling
        ruling_input = os.path.join(input_folder, ruling)  # Changed variable name
        if not os.path.exists(ruling_input):
            print(f"Error: Ruling folder not found - {ruling_input}")  # Updated error message
            continue
            
        for root, _, files in os.walk(ruling_input):
            midi_files = [f for f in files if f.endswith('.mid')]
            if not midi_files:
                continue
                
            for file in midi_files:
                midi_path = os.path.join(root, file)
                
                # Get relative path and create output directories
                rel_path = os.path.relpath(root, ruling_input)
                abs_output_dir = os.path.join(abs_output_folder, ruling, rel_path)  # Changed category to ruling
                rel_output_dir = os.path.join(rel_output_folder, ruling, rel_path)  # Changed category to ruling
                
                os.makedirs(abs_output_dir, exist_ok=True)
                os.makedirs(rel_output_dir, exist_ok=True)
                
                file_info = get_file_info(midi_path, os.path.join(input_folder, ruling))  # Changed category to ruling
                process_file(midi_path, abs_output_dir, rel_output_dir, file[:-4], library, file_info, window_size, step_size)
    
    # Store both versions of the library
    store_library(library_path, library)  # Original segmented version
    unsegmented_path = library_path.replace("melody_library.csv", "melody_library_unsegmented.csv")
    store_library_unsegmented(unsegmented_path, library)
    
    # Print summary
    print(f"\nProcessing completed with:")
    print(f"- Window size: {window_size}")
    print(f"- Step size: {step_size}")
    print(f"Segmented library saved to: {library_path}")
    print(f"Unsegmented library saved to: {unsegmented_path}")

def check_directories():
    """Checks if required directories exist and creates them if necessary."""
    dataset_folder = os.path.join(os.path.dirname(__file__), "MCIC_Dataset")
    input_folder = os.path.join(dataset_folder, "MCIC_Raw")
    
    # Create paths for Plagiarism and No_Plagiarism folders
    plagiarism_input = os.path.join(input_folder, "Plagiarism")
    no_plagiarism_input = os.path.join(input_folder, "No_Plagiarism")
    
    directories = {
        'dataset': dataset_folder,
        'input': input_folder,
        'input_plagiarism': plagiarism_input,
        'input_no_plagiarism': no_plagiarism_input
    }
    
    missing = []
    for name, path in directories.items():
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print("Error: Required input directories not found:")
        for path in missing:
            print(f"  - {path}")
        print("\nPlease ensure the following directory structure exists:")
        print(f"MelDet/")
        print(f"└── MCIC_Dataset/")
        print(f"    ├── MCIC_Raw/")
        print(f"    │   ├── Plagiarism/")
        print(f"    │   │   ├── Case1/")
        print(f"    │   │   │   ├── file1.mid")
        print(f"    │   │   │   └── file2.mid")
        print(f"    │   │   └── Case2/...")
        print(f"    │   └── No_Plagiarism/")
        print(f"    │       ├── Case3/")
        print(f"    │       │   ├── file1.mid")
        print(f"    │       │   └── file2.mid")
        print(f"    │       └── Case4/...")
        print(f"    └── MCIC_Preprocessed/")
        exit(1)
    
    return input_folder

if __name__ == "__main__":
    # Define the root directories using the correct path structure
    dataset_folder = os.path.join(os.path.dirname(__file__), "MCIC_Dataset")
    input_folder = os.path.join(dataset_folder, "MCIC_Raw")
    output_folder = os.path.join(dataset_folder, "MCIC_Preprocessed")
    
    # Check if required directories exist
    input_folder = check_directories()
    
    # Process all files in all subfolders
    process_folder(
        input_folder,
        os.path.join(output_folder, "absolute_sequences"),
        os.path.join(output_folder, "relative_intervals"),
        os.path.join(output_folder, "melody_library.csv")
    )
    
    print("\nProcessing completed.")
    print(f"Preprocessed data saved in {output_folder}")