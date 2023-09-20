for i in __import__("itertools").accumulate(__import__("itertools").count(1), func=lambda a, b: a + ord('Close'[b % 5]), initial=67):
    print(i)
    if i > 900:
        break

'''
o= 0
for number in [sum:=sum+ord('Close'[i %5]) for i in __import__("itertools").count(0)]:
    print(number)
    o+=1
    if o > 20:
        break
'''

