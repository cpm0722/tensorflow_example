1. multi variable linear regression에서 cost function 작성 시 reduce_mean과 reduce_sum 사이의 차이점
  
  -> lab04-1.py에서 두 코드의 결과값이 다름 (reduce_mean은 정상적인 결과값, reduce_sum은 nan값)
 
 -> mean과 sum은 상수 값으로 나눈 다는 점에서만 차이가 있기에 cost function의 역할에서의 차이가 있지는 않을텐데, 왜 결과값은 다른지?
