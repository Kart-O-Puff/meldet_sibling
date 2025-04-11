# meldet_sibling
repository nila FeBiLair 👩🏻‍🎓👨🏻‍🎓👩🏻‍🎓

# How to run

1. pip install -r "requirements.txt"
2. run the data_preprocess_2.py
3. run the meldet_approach.py, sumcommon_approach.py and tversky_approach.py
### you may choose to visualize how the approaches generated similarity scores in the user menu
### you may also choose to print the result in table form without visualization
### you may also view the similarity_report_(approach used).csv that stored the computed similarity scores of each approach used
4. run the evaluation.py
### you may choose to visualize the results of the evaluation metrics with graph representations
5. repeat steps 2 to 4 by testing the three approaches on different window size and step size

## Revisions Made:
- There is a major change on what python library was used for preprocessing the midi files. Instead of using pretty_midi, we now use music21 since it has better capabilities on expressing rhythmic elements in the form of standard note duration unlike pretty_midi which can only give the rhythm sequence in the form of seconds or time (which greatly impacts the computation for getting relative rhythm sequence).
- We now compare our proposed approach with Sum Common with Jaccard-like Normalization, the Original Tversky Measure and Hungarian Approach for similarity scoring. 

## Mga Kailangan pang gawin:
- Maglagay pa ng test cases sa MCIC_Dataset > MCIC_Raw para matest ang performance ng similarity scoring ng mga approach sa iba't ibang case.
- I-test at evaluate ang tatlong approach sa iba't ibang window size at step size.
