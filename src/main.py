from learner.train import train_model, load_and_predict

class C:
    TRAIN = False

def main(f1, f2):
    # train the model if needed, it is unnneeded in prod env
    if C.TRAIN: 
        train_model()
    
    # predict r1, r2, r3, r4, r5 values on given f1 and f2 values by using trained models
    result = load_and_predict(f1, f2) # return a list that contains 5 float numbers
    
    return result
    
if __name__ == '__main__':
    f1 = 10
    f2 = 60
    print(main(f1, f2))