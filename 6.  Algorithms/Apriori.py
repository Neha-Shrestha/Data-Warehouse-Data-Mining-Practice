from collections import defaultdict
from itertools import combinations

def get_frequent_1_itemsets(transactions, min_support):
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    
    frequent_1_itemsets = {frozenset([item]): count for item, count in item_counts.items() if count >= min_support}
    return frequent_1_itemsets

def generate_candidate_itemsets(prev_itemsets, length):
    candidates = set()
    for itemset1 in prev_itemsets:
        for itemset2 in prev_itemsets:
            if len(itemset1.union(itemset2)) == length:
                candidates.add(itemset1.union(itemset2))
    return candidates

def get_frequent_itemsets(transactions, min_support):
    frequent_itemsets = []
    frequent_1_itemsets = get_frequent_1_itemsets(transactions, min_support)
    frequent_itemsets.append(frequent_1_itemsets)
    k = 2
    
    while True:
        candidate_itemsets = generate_candidate_itemsets(frequent_itemsets[k-2].keys(), k)
        item_counts = defaultdict(int)
        for transaction in transactions:
            for candidate in candidate_itemsets:
                if candidate.issubset(transaction):
                    item_counts[candidate] += 1
        
        frequent_k_itemsets = {itemset: count for itemset, count in item_counts.items() if count >= min_support}
        if frequent_k_itemsets:
            frequent_itemsets.append(frequent_k_itemsets)
            k += 1
        else:
            break
    
    return frequent_itemsets

def calculate_confidence(frequent_itemsets, antecedent, consequent):
    antecedent_set = frozenset(antecedent)
    antecedent_support = frequent_itemsets[len(antecedent_set) - 1][antecedent_set]
    
    consequent_set = frozenset(consequent)
    rule_support = frequent_itemsets[len(antecedent_set) + len(consequent_set) - 1][antecedent_set.union(consequent_set)]
    
    return rule_support / antecedent_support

def main():
    transactions = [
        {"M", "O", "N", "K", "E", "Y"},
        {"D", "O", "N", "K", "E", "Y"},
        {"M", "A", "K", "E"},
        {"M", "U", "C", "K", "Y"},
        {"C", "O", "O", "K", "I", "E"}
    ]
    min_support = 3
    frequent_itemsets = get_frequent_itemsets(transactions, min_support)
    for k, itemsets in enumerate(frequent_itemsets, start=1):
        print(f"\nFrequent {k}-itemsets:")
        for itemset, support in itemsets.items():
            print(f"{itemset}: {support}")
    
    print("\nAssociation rules with confidence:")
    for itemsets in frequent_itemsets[2:]:
        for itemset in itemsets:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    consequent = itemset - set(antecedent)
                    confidence = calculate_confidence(frequent_itemsets, set(antecedent), consequent)
                    print(f"{antecedent} -> {consequent}: {confidence}")

if __name__ == "__main__":
    main()