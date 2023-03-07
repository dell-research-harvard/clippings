###Test script

def print_message():
    print(args.message)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', type=str, default='Hello World')
    args = parser.parse_args()
    print_message()

    