# Machine_Comprehension_with_external_knowledge
Worked done during my internship at Kyoto university,Japan

For running the code for machine Comprehension with external knowledge- Kyoto university Intern

1.All the scripts are provided

2.Do the pre-processing of Dataset you wish to use: CNN NewsQA or SQuAD1.1

3.Name the files accordingly while doing pre-processing

4.Create the external knowledge from script

  e.g like wordnet-synonyms.txt
  
5.make the label file you wish to used e.g syn_labelset.txt

6.After all the pre-processing is done run this:

  ./training_kim_bidaf.sh -g (gpu) -e (epochs) -l (no. of total epochs) -c (config file) -o (output directory) -n (output file name)
  
7.Wait for results

8.one epoch take around 45 mins.

9.Thank You
