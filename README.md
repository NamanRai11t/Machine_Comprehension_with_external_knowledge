# Machine_Comprehension_with_external_knowledge
Worked done during my internship at Kyoto university,Japan
For running the code for machine Comprehension with external knowledge- Kyoto university Intern

All the scripts are provided
Do the pre-processing of Dataset you wish to use: CNN NewsQA or SQuAD1.1
Name the files accordingly while doing pre-processing
Create the external knowledge from script e.g like wordnet-synonyms.txt
make the label file you wish to used e.g syn_labelset.txt
After all the pre-processing is done run this: ./training_kim_bidaf.sh -g (gpu) -e (epochs) -l (no. of total epochs) -c (config file) -o (output directory) -n (output file name)
Wait for results
one epoch take around 45 mins.
Thank You
