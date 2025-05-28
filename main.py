from scripts.inference import infer

def main():
    print('Share your thoughts')
    while True:
        text = input('> ')
        print('>', infer(text))

if __name__ == '__main__':
    main()