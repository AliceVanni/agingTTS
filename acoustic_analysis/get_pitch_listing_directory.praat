#Based on 4.7.2003 Mietta Lennes' script for getting pitch maximum

# script goes through sound files in a directory,
# opens each Sound, calculates pitch 
# and saves results to a text file.


form Analyze pitch in files
  comment Directory of sound files
  text sound_directory data
  comment Full path of the resulting text file:
  text resultfile pitchresults_bn.txt
  comment Pitch analysis parameters
  positive Time_step 0.01
  positive Minimum_pitch_(Hz) 20
  positive Maximum_pitch_(Hz) 450
endform

# Here, you make a listing of all the sound files in a directory.
# The example gets file names ending with ".wav" from data/

Create Strings as file list... list 'sound_directory$'/*.wav
numberOfFiles = Get number of strings

# Check if the result file exists:
if fileReadable (resultfile$)
  pause The result file 'resultfile$' already exists! Do you want to overwrite it?
  filedelete 'resultfile$'
endif

# Write a row with column titles to the result file:
# (remember to edit this if you add or change the analyses!)

titleline$ = "Filename  Time stamp  Pitch (Hz)'newline$'"
fileappend "'resultfile$'" 'titleline$'

# Go through all the sound files, one by one:

for ifile to numberOfFiles
  filename$ = Get string... ifile
  # A sound file is opened from the listing:
  Read from file... 'sound_directory$'/'filename$'
  # Starting from here, you can add everything that should be 
  # repeated for every sound file that was opened:
  soundname$ = selected$ ("Sound", 1)
  To Pitch... time_step minimum_pitch maximum_pitch
      select Pitch 'soundname$'
      no_of_frames = Get number of frames
            for frame from 1 to no_of_frames
                time = Get time from frame number: frame
                pitch = Get value in frame: frame, "Hertz"
                resultline$ = "'soundname$'  'time'  'pitch''newline$'"
                fileappend "'resultfile$'" 'resultline$'
            endfor

  # Remove the temporary objects from the object list
  select Sound 'soundname$'
  plus Pitch 'soundname$'
  Remove
  select Strings list
  # and go on with the next sound file!
endfor

Remove
