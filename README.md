# A simple speech synthesizer

The synthesizer use concatenation of monophones

The corpus used is cmu pronouncing dictionary (English)

* before run, please use nltk
	`python -m pip install nltk`

	`python`

	`>>> import nltk`

	`>>> import nltk.download()`, to download all nltk related corpus
	
* To run use:
	`python synthesizer.py "message to be synthesized" --monophones "dir of monophones" -v 1.0 -p -o "outfile.wav"`