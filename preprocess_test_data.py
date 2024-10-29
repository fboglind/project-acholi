from subword_nmt.apply_bpe import BPE

# Create BPE processor using existing codes
with open('onmt_data/data.ach.codes', 'r', encoding='utf-8') as codes_file:
    bpe = BPE(codes_file)

# Apply BPE to test file
with open('processed_data_moses/salt.test.tk.lc.ach', 'r', encoding='utf-8') as infile, \
     open('onmt_data/test.bpe.ach', 'w', encoding='utf-8') as outfile:
    for line in infile:
        outfile.write(bpe.process_line(line))
print("file saved: onmt_data/test.bpe.ach")