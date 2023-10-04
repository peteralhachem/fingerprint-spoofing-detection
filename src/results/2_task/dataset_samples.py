from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set, \
    load_fingerprint_spoofing_detection_test_set

if __name__ == '__main__':

    train_set, train_labels = load_fingerprint_spoofing_detection_train_set()
    eval_set, eval_labels = load_fingerprint_spoofing_detection_test_set()


    print("* training set *")

    print(f"# samples for the authentic (true) class: "
          f"{train_set[:, train_labels == 1].shape[1]}")
    print(f"# samples for the spoofed (false) class: "
            f"{train_set[:, train_labels == 0].shape[1]}")


    print("\n* evaluation set * ")

    print(f"# samples for the authentic (true) class: "
          f"{eval_set[:, eval_labels == 1].shape[1]}")
    print(f"# samples for the spoofed (false) class: "
            f"{eval_set[:, eval_labels == 0].shape[1]}")
