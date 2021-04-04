import os   
import json


def data_match(formula_list, num, split_road, k):
    for i in range(num):
        formula_dict = {"formula_road": "", "formula_str": ""}
        formula_idx = split_road + str(i) + "/"
        # formula_idx need to be modified
        if os.path.exists(formula_idx + "formula.txt"):
            with open(formula_idx + "formula.txt", "r", encoding="ISO-8859-1") as f:
                formula = f.readlines()
                formula_dict["formula_str"] = formula
            formula_dict["formula_road"] = formula_idx + "formula.jpg"
            formula_list.append(formula_dict)
        else:
            print(formula_idx + " is Null! PASS!")
            k += 1
    return formula_list, k


def main(split):
    count = []
    formula_list = []
    split_road = "/home/hhhfccz/im2latex/dataset/" + split + "/"
    # split_road need to be modified
    [count.append(int(idx)) for idx in os.listdir(split)]
    # print(sorted(count)[-1])
    formula_list, k = data_match(formula_list, num=sorted(count)[-1], split_road=split_road, k=0)
    # print(k)
    formula_list = json.dumps(formula_list, indent=4)
    with open("annotations_" + split + ".json", "w", encoding="ISO-8859-1") as file:
        file.write(formula_list)
    print("Annotations of " + split + " is OK!")


if __name__ == "__main__":
    splits = ["train", "test", "validate"]
    for split in splits:
        print(split)
        main(split=split)
