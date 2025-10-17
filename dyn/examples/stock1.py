ccc=closes
print(ccc)
X=[]
p1=ccc[0]
p2=ccc[1]
p3=ccc[2]
for p in ccc[3:]:
  X.append([p1,p2,p3])
  p1=p2
  p2=p3
  p3=p
print(X) #入力
y=ccc[4:]
print(y) #出力?
