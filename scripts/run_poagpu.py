import sys
import os
import subprocess

def print_help():
    print("""
Uso corretto:
    python scripts/run_poagpu.py <numBlocks> <num_vertici> <num_reads> <len_reads> [--example]

Dove:
    <num_vertici>           = Numero di vertici (presente nel nome del file grafo).
    <num_reads>             = Numero di reads (valore numerico, verrà convertito in 100, 100K, 100M automaticamente).
    <len_reads>             = Lunghezza delle reads (presente nel nome del file delle reads).
    --example               = (Opzionale) Usa i file di esempio da test/examples/.
""")

def format_reads_folder(num_reads):
    if num_reads >= 1_000_000:
        return f"{num_reads // 1_000_000}M"
    elif num_reads >= 1_000:
        return f"{num_reads // 1_000}K"
    else:
        return str(num_reads)

def main():
    if len(sys.argv) not in [5, 6]:
        print("[ERRORE] Numero errato di parametri.")
        print_help()
        sys.exit(1)

    use_example = False
    if len(sys.argv) == 6:
        if sys.argv[5] == "--example":
            use_example = True
        else:
            print("[ERRORE] Opzione non riconosciuta.")
            print_help()
            sys.exit(1)

    try:
        numBlocks = int(sys.argv[1])
        num_vertici = int(sys.argv[2])
        num_reads = int(sys.argv[3])
        len_reads = int(sys.argv[4])
    except ValueError:
        print("[ERRORE] Tutti i parametri devono essere numeri interi.")
        print_help()
        sys.exit(1)
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

      # Percorsi relativi
    if use_example:
        graph_file_rel = "test/examples/graph.gfa"
        reads_file_rel = "test/examples/reads.fa"
    else:
        reads_folder = format_reads_folder(num_reads)
        graph_file_rel = f"test/graph/graph_{num_vertici}.gfa"
        reads_file_rel = f"test/reads/{reads_folder}/reads_{len_reads}.fa"

    # Percorsi assoluti per il controllo
    graph_file_abs = os.path.join(base_dir, graph_file_rel)
    reads_file_abs = os.path.join(base_dir, reads_file_rel)
    poagpu_exec = os.path.join(base_dir, "poagpu")

    # Controllo esistenza file
    if not os.path.isfile(poagpu_exec):
        print(f"[ERRORE] Eseguibile non trovato: {poagpu_exec}")
        sys.exit(1)
    if not os.path.isfile(graph_file_abs):
        print(f"[ERRORE] File grafo non trovato: {graph_file_abs}")
        sys.exit(1)
    if not os.path.isfile(reads_file_abs):
        print(f"[ERRORE] File reads non trovato: {reads_file_abs}")
        sys.exit(1)

    # Comando da eseguire
    cmd = ["./poagpu", str(numBlocks), reads_file_rel, graph_file_rel]

    print(f"Eseguo comando: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERRORE] Errore durante l'esecuzione: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()