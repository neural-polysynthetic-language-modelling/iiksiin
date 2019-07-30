import os
import argparse

def preprocess_fst(in_file, out_file):
	assert os.path.exists(in_file)
	print("Opening the fst analyzed file in ", in_file)
	with open(in_file, "r") as fst_analysis:
		with open(out_file, "w") as fst_preprocessed:
			for line in fst_analysis:
				tokens = line.split()
                                n_tokens = len(tokens)
                                token_idx = 0
				for token in tokens:
					if token.startswith("*"):
						new_token = token[1:] # remove *
						new_character_token = '^'.join(list(new_token)) # tokenize it by character 
						new_character_token = new_character_token.replace("*", "") # remove extra asterisks
                                                if token_idx == n_tokens-1:
                                                    fst_preprocessed.write(new_token)
                                                else:
                                                    fst_preprocessed.write(new_token)
                                               # pass
					else:
						new_token = token.split(":")[-2] # get the fst segmentation 
						new_token = new_token.replace("*", "") # remove extra asterisks
						fst_preprocessed.write(new_token + " _ ")

				fst_preprocessed.write("\n")

	print("Saved the preprocessed file in ", out_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_path', '-ip', type=str, required=True, 
	help= 'path to the file with fst output e.g. ess_fst/all.train.analyzed.ess')
	parser.add_argument('--out_path', '-op', type=str, required=True, 
	help= 'path to the output file e.g. ess_fst/all.train.analyzed.preprocessed.ess')
	args = parser.parse_args()
	preprocess_fst(args.in_path, args.out_path)
