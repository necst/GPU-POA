import random
from pathlib import Path

def random_nucleotide(exclude=None):
    nucleotides = list("ACGT")
    if exclude:
        nucleotides.remove(exclude)
    return random.choice(nucleotides)

def introduce_snps(seq, num_snps):
    snp_positions = sorted(random.sample(range(1, len(seq) - 2), num_snps))
    snps = {}
    for pos in snp_positions:
        original = seq[pos]
        variant = random_nucleotide(exclude=original)
        snps[pos] = variant
    return snps

def generate_gfa(seq, snps):
    segments = []
    links = []
    
    current_seq_id = 1
    last_pos = 0
    last_segments = []

    for pos, variant in snps.items():
        # Segment before SNP
        if pos > last_pos:
            segment_seq = seq[last_pos:pos]
            segment_id = f"seq{current_seq_id}"
            segments.append((segment_id, segment_seq))
            if last_segments:
                for lsid in last_segments:
                    links.append((lsid, segment_id))
            last_segments = [segment_id]
            current_seq_id += 1

        # Create two branches: variant and original
        original_base = seq[pos]

        variant_id = f"seq{current_seq_id}"
        segments.append((variant_id, variant))
        current_seq_id += 1

        original_id = f"seq{current_seq_id}"
        segments.append((original_id, original_base))
        current_seq_id += 1

        # Both branches link from previous segment
        for lsid in last_segments:
            links.append((lsid, variant_id))
            links.append((lsid, original_id))

        # Create a common segment after SNP (1bp to reconnect)
        merge_base = seq[pos + 1]
        merge_id = f"seq{current_seq_id}"
        segments.append((merge_id, merge_base))
        current_seq_id += 1

        links.append((variant_id, merge_id))
        links.append((original_id, merge_id))

        last_segments = [merge_id]
        last_pos = pos + 1 + 1  # move past SNP and merge base

    # Remaining sequence after last SNP and merge
    if last_pos < len(seq):
        segment_seq = seq[last_pos:]
        if segment_seq:
            segment_id = f"seq{current_seq_id}"
            segments.append((segment_id, segment_seq))
            for lsid in last_segments:
                links.append((lsid, segment_id))

    # Generate GFA content
    gfa_lines = ["H\tVN:Z:1.0"]
    for seg_id, seg_seq in segments:
        gfa_lines.append(f"S\t{seg_id}\t{seg_seq}")
    for from_seg, to_seg in links:
        gfa_lines.append(f"L\t{from_seg}\t+\t{to_seg}\t+\t*")

    return "\n".join(gfa_lines)

def main():
    # Sequenza DNA fornita
    original_seq = "AATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGTGTTCGAGGGCTAGCTCGAAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGTGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCAATCTGTTCGAGGGCTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGAAATCTGTTCGAGGGCTAGCTCGATGTAGCTCGATGATTGCGTCGTGGGAAAAATCTCAGA"

    num_snps = 20
    snps = introduce_snps(original_seq, num_snps)
    gfa_content = generate_gfa(original_seq, snps)

    output_file = Path("../test/graph/graph_1000.gfa")
    with output_file.open("w") as f:
        f.write(gfa_content)

if __name__ == "__main__":
    main()






