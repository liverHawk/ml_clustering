from itertools import combinations

attack_labels = ["Heartbleed","Infiltration","Bot","DoS","DDoS","SSH-Patator","PortScan","FTP-Patator","Web Attack"]

# クラスタリングの組み合わせを作成
# 2 <= unknown_labels <= attack_labels
# -> 0 <= known_labels <= attack_labels - 2

with open("queue.sh", "w") as f:
    for len_known_labels in range(0, len(attack_labels) - 3):
        known_labels = combinations(attack_labels, len_known_labels)
        for known_labels in known_labels:
            labels_str = str(list(known_labels)).replace("'", '"')
            f.write(f"dvc exp run -S known_labels='{labels_str}' --queue\n")

