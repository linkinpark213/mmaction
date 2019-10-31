import os
import random


def divide_splits(classes):
    train_file = open('annotations/trainlist01.txt', 'w+')
    test_file = open('annotations/testlist01.txt', 'w+')
    for i, clazz in enumerate(classes):
        seqs = list(os.listdir('videos/' + clazz))
        seqs.sort()
        for seq in seqs:
            temp = random.random()
            if temp > 0.75:
                test_file.write('{}/{} {}\n'.format(clazz, seq, i+1))
            else:
                train_file.write('{}/{} {}\n'.format(clazz, seq, i+1))
    test_file.close()
    train_file.close()


def generate_class_indices(classes):
    inds_txt = open('annotations/classInd.txt', 'w+')
    
    for i, clazz in enumerate(classes):
        inds_txt.write('{} {}\n'.format(i+1, clazz))
    
    inds_txt.close()

if __name__ == '__main__':
    if not os.path.isdir('annotations'):
        os.mkdir('annotations')
    
    classes = list(os.listdir('videos'))
    classes.sort()
    
    generate_class_indices(classes)
    divide_splits(classes)
