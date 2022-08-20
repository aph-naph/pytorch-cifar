def input_string1():
    epoch = int(input().split()[1]) + 1
    epoch = 'Epoch: ' + str(epoch)
    train = '    Train - ' + '|'.join(input().split('|')[-2:])
    val = '    Val   - ' + '|'.join(input().split('|')[-2:])
    input()
    input()
    test = 'Test set - ' + '|'.join(input().split('|')[-2:])
    print('\n'.join([epoch, train, val, test]))

def input_string2():
    epoch = int(input().split()[1]) + 1
    epoch = 'Epoch: ' + str(epoch)
    train = '    Train - ' + '|'.join(input().split('|')[-2:])
    input()
    val = '    Val   - ' + '|'.join(input().split('|')[-2:])
    input()
    input()
    input()
    test = 'Test set - ' + '|'.join(input().split('|')[-2:])
    print('\n'.join([epoch, train, val, test]))

input_string1()
# input_string2()

""" Input string 1
Epoch: 19
 [============================ 422/422 ===========================>]  Step: 39ms | Tot: 19s779ms | Loss: 0.001 | Acc: 100.000% (54000/54000)                                                         
 [============================ 47/47 ============================>.]  Step: 10ms | Tot: 579ms | Loss: 0.016 | Acc: 99.567% (5974/6000)                                                               

Evaluating on test set...
 [============================ 79/79 =============================>]  Step: 12ms | Tot: 949ms | Loss: 0.013 | Acc: 99.570% (9957/10000)
"""

""" Input string 2
Epoch: 49                                                                                                                                                                                            
 [============================ 352/352 ===========================>]  Step: 34ms | Tot: 21s18ms | Loss: 0.011 | Acc: 99.813% (44916/45000)                                                           
                                                                                                                                                                                                     
 [============================ 40/40 ============================>.]  Step: 2ms | Tot: 619ms | Loss: 0.282 | Acc: 92.120% (4606/5000)                                                                
                                                                                                                                                                                                     
                                                                                                                                                                                                     
Evaluating on test set...                                                                                                                                                                            
 [============================ 79/79 =============================>]  Step: 14ms | Tot: 1s243ms | Loss: 0.295 | Acc: 91.730% (9173/10000)
 """