"""Useful dictionaries to have around for bioinformatics."""

aa = {'A': 'Ala',
      'R': 'Arg',
      'N': 'Asn',
      'D': 'Asp',
      'C': 'Cys',
      'Q': 'Gln',
      'E': 'Glu',
      'G': 'Gly',
      'H': 'His',
      'I': 'Ile',
      'L': 'Leu',
      'K': 'Lys',
      'M': 'Met',
      'F': 'Phe',
      'P': 'Pro',
      'S': 'Ser',
      'T': 'Thr',
      'W': 'Trp',
      'Y': 'Tyr',
      'V': 'Val'}

# The set of DNA bases
bases = ['T', 'C', 'A', 'G']

# Build list of codons
codon_list = []
for first_base in bases:
    for second_base in bases:
        for third_base in bases:
            codon_list += [first_base + second_base + third_base]

# The amino acids that are coded for (* = STOP codon)
amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'

# Build dictionary from tuple of 2-tuples (technically an iterator, but it works)
codons = dict(zip(codon_list, amino_acids))

