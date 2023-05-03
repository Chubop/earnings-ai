import time

current_words = []


def print_text_in_chunks(file_path, split_size=20):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split(" ")

    for i in range(0, len(words)):
        word = words[i].replace('\n', ' ')
        if i == 0:
            print(word, end=" ")
        else:
            print(word, end="\n" if i % split_size == 0 else " ")
        current_words.append(word)
        time.sleep(1 / split_size)


print_text_in_chunks('transcripts/united_health_transcript')
