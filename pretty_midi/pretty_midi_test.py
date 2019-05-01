from pretty_midi import PrettyMIDI 
import numpy as np
NUM_VELOCITIES = 128
def main():
    file_name = "midi_files/reformated/clementi/clementi_opus36_3_1_format0.mid"
    midi = PrettyMIDI(file_name)
    midi.remove_invalid_notes()
    for instrument in midi.instruments:
        roll = instrument.get_piano_roll(fs=4).T
        orig_roll = roll
        # trim beginning silence
        summed = np.sum(roll, axis=1)
        mask = (summed > 0).astype(float)
        roll = roll[np.argmax(mask):]
        # transform note velocities into 1s
        roll_w_velocity = roll
        roll = (roll > 0).astype(float)
        # get the notes and their velocity
        I = np.nonzero(roll)
        notes = []
        velocities = []
        for i in range(len(I[0])):
            frame = I[0][i]
            note = I[1][i]
            velocity = int(roll_w_velocity[frame][note])
            notes.append(note)
            velocities.append(velocity)

        r0 = roll[0]
        v0 = roll_w_velocity[0]
        r_i = np.nonzero(r0)[0]
        v_i = np.nonzero(v0)[0]
        RI = np.nonzero(roll)
        rows = RI[0]
        cols = RI[1]
        VI = np.nonzero(roll_w_velocity)
        v_rows = VI[0]
        v_cols = VI[1]

        #note_index = np.nonzero(roll)[0]
        #note_num = note_index
        #velocity = velocities[note_index]

        # append a feature: 1 to rests and 0 to notes
        rests = np.sum(roll, axis=1)
        rests = (rests != 1).astype(float)
        roll = np.insert(roll, 0, rests, axis=1)
        r = roll[0]


if __name__ == '__main__':
    main()