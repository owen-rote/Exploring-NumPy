## Exploring basic and practical NumPy functionalities
**[github.com/owen-rote](https://github.com/owen-rote)**
* **Indexing:** Accessing and manipulating elements within arrays
* **Mathematics:** Arithmetic operations, trigonometric functions, etc
* **Linear Algebra:** Matrix & vector operations, solving systems of equations
* **Statistics:** Calculating common statistics from datasets
## Followed: [Tutorial by freeCodeCamp.org on Youtube](https://www.youtube.com/watch?v=QUT1VHiLmmI)


```python
import numpy as np
```

# Basics


```python
a = np.array([1,2,3])
a

# a = np.array([1, 2, 3]), dtype='int32')
```




    array([1, 2, 3])




```python
b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
print(b)
```

    [[9. 8. 7.]
     [6. 5. 4.]]
    


```python
# Get Dimension
b.ndim
```




    2




```python
# Get Shape
print(b.shape)
print("(Length, Height)")
```

    (2, 3)
    (Length, Height)
    


```python
# Get Type
b.dtype
```




    dtype('float64')




```python
# Get Size (Number of bytes for each element)
b.itemsize
```




    8




```python
# Get Total Size (# of bytes)
b.nbytes
```




    48



# Accessing/Changing specific elements, rows, columns, etc


```python
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
a.shape # 2*7 array
```

    [[ 1  2  3  4  5  6  7]
     [ 8  9 10 11 12 13 14]]
    




    (2, 7)




```python
# Get specific element [r, c]
a[1, 5]
a[1, -2] # Same thing
```




    13




```python
# Get a specific row
a[0, :]
```




    array([1, 2, 3, 4, 5, 6, 7])




```python
# Get a specific column
a[:, 2]
```




    array([ 3, 10])




```python
# More interesting indexing
# [start_index : end_index : step_size]
a[0, 1 : -1 : 2]
```




    array([2, 4, 6])




```python
# Change element at index
a[1,5] = 20
a
```




    array([[ 1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 20, 14]])




```python
# Replacing a row with 5's
a[:,2] = 5
print(a)

# Replacing a row with a sequence
a[:,2] = [1, 2]
print(a)
```

    [[ 1  2  5  4  5  6  7]
     [ 8  9  5 11 12 20 14]]
    [[ 1  2  1  4  5  6  7]
     [ 8  9  2 11 12 20 14]]
    


```python
# 3D Example
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)
```

    [[[1 2]
      [3 4]]
    
     [[5 6]
      [7 8]]]
    


```python
# Get specific element (work outside->in)
b[0,1,1]

# First block: 0
# Second row: 1
# Second index: 1
# [First layer, Second row, Second index)
```




    4




```python
b[:,1,:]

# [Both layers, second row, both indexes)
```




    array([[3, 4],
           [7, 8]])




```python
# Replacing
b[:,1,:] = [[9,9],[8,8]]
b
```




    array([[[1, 2],
            [9, 9]],
    
           [[5, 6],
            [8, 8]]])



# Initializing Different Types of Arrays


```python
# All 0's matrix
np.zeros((2,3))
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
# All 1's matrix
np.ones((4,2,2), dtype='int32') # dtype optional
```




    array([[[1, 1],
            [1, 1]],
    
           [[1, 1],
            [1, 1]],
    
           [[1, 1],
            [1, 1]],
    
           [[1, 1],
            [1, 1]]])




```python
# Any other number
np.full((2,2), 99)
# 2*2 with all 99's
```




    array([[99, 99],
           [99, 99]])




```python
# Any other number (full_like)
np.full_like(a, 4)
# Copies dimensions of a
```




    array([[4, 4, 4, 4, 4, 4, 4],
           [4, 4, 4, 4, 4, 4, 4]])




```python
# Random decimal number matrix
np.random.rand(4, 2, 3)
# np.random.random_sample(a.shape) Uses dimensions of a
```




    array([[[0.24314925, 0.30547436, 0.394195  ],
            [0.81826472, 0.85545752, 0.33905719]],
    
           [[0.42945287, 0.04829369, 0.22161343],
            [0.58728573, 0.2513226 , 0.56636948]],
    
           [[0.56794331, 0.95565567, 0.74505303],
            [0.48501734, 0.45798407, 0.00117981]],
    
           [[0.93605788, 0.13317905, 0.99469853],
            [0.74652986, 0.34234903, 0.92084808]]])




```python
# Random Integer values
np.random.randint(-4,7, size=(3,3))
```




    array([[-2, -1,  4],
           [ 2,  5, -4],
           [-4,  0,  5]])




```python
# Identity Matrix
np.identity(5)
```




    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])




```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
r1 = np.repeat(arr, 3, axis=0) # axis = 0 or 1
print(r1)
```

    [[1 2 3]
     [1 2 3]
     [1 2 3]
     [4 5 6]
     [4 5 6]
     [4 5 6]]
    

#### Making
#### 1 1 1 1 1
#### 1 0 0 0 1
#### 1 0 9 0 1
#### 1 0 0 0 1
#### 1 1 1 1 1


```python
output = np.ones((5, 5))
print(output, end="\n\n")

z = np.zeros((3, 3))
z[1, 1] = 9
print(z, end="\n\n")

output[1:4, 1:4] = z
print(output)
```

    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    
    [[0. 0. 0.]
     [0. 9. 0.]
     [0. 0. 0.]]
    
    [[1. 1. 1. 1. 1.]
     [1. 0. 0. 0. 1.]
     [1. 0. 9. 0. 1.]
     [1. 0. 0. 0. 1.]
     [1. 1. 1. 1. 1.]]
    

##### Be careful when copying arrays!!!


```python
a = np.array([1, 2, 3])
b = a.copy() # Use copy() or it will make b simply point to a
b[0] = 100
print(b)
print(a)
```

    [100   2   3]
    [1 2 3]
    

# Mathematics


```python
a = np.array([1, 2, 3, 4])
print(a)
```

    [1 2 3 4]
    


```python
print(a + 2)
print(a - 2)
print(a * 2)
print(a / 2)
a += 2
a
```

    [3 4 5 6]
    [-1  0  1  2]
    [2 4 6 8]
    [0.5 1.  1.5 2. ]
    




    array([3, 4, 5, 6])




```python
# Take the sin
np.sin(a)
```




    array([ 0.14112001, -0.7568025 , -0.95892427, -0.2794155 ])



https://docs.scipy.org/doc/numpy/reference/routines.math.html

# Linear Algebra

##### Multiplying Matrices


```python
a = np.ones((2, 3))
print(a)

b = np.full((3, 2), 2)
print(b)

np.matmul(a, b)
```

    [[1. 1. 1.]
     [1. 1. 1.]]
    [[2 2]
     [2 2]
     [2 2]]
    




    array([[6., 6.],
           [6., 6.]])




```python
c = np.identity(3)
np.linalg.det(c) # Find determinant
```




    1.0



# Statistics


```python
stats = np.array([[1, 2, 3], [4, 5, 6]])
stats
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
np.min(stats, axis=1) # [Min of first row, Min of second]
```




    array([1, 4])




```python
np.max(stats)
```




    6




```python
np.sum(stats) # all elements
np.sum(stats, axis=0) # Sum of each col
```




    array([5, 7, 9])



# Reorganizing Arrays


```python
before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(before)

after = before.reshape((2, 2, 2))
print(after)
```

    [[1 2 3 4]
     [5 6 7 8]]
    [[[1 2]
      [3 4]]
    
     [[5 6]
      [7 8]]]
    


```python
# Vertically Stacking Vectors
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])

np.vstack([v1, v2, v2, v2])
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8],
           [5, 6, 7, 8],
           [5, 6, 7, 8]])




```python
# Horizontally Stacking
h1 = np.ones((2, 4))
h2 = np.zeros((2, 2))

np.hstack((h1, h2))
```




    array([[1., 1., 1., 1., 0., 0.],
           [1., 1., 1., 1., 0., 0.]])



# Misc

##### Load data from file


```python
#filedata = np.genfromtxt('data.txt', delimiter=',')
#filedata.astype('int32') # Copies data into specified type
#filedata = filedata.astype('int32')
filedata = np.array([[24, 675, 345, 4, 235, 35, 99, 203], 
                     [6, 645, 4, 0, 41, 456, 53, 25]])
```

##### Bool Masking and Advanced Indexing


```python
filedata > 50
# Compares each element to 50. Makes array of true, false, etc
```




    array([[False,  True,  True, False,  True, False,  True,  True],
           [False,  True, False, False, False,  True,  True, False]])




```python
filedata[filedata > 50]
# Grabs all values greater than 50
```




    array([675, 345, 235,  99, 203, 645, 456,  53])




```python
np.any(filedata > 50, axis=0)
# What columns have a value > 50
```




    array([False,  True,  True, False,  True,  True,  True,  True])




```python
np.all(filedata > 50, axis=0)
# Which columns have all values > 50
```




    array([False,  True, False, False, False, False,  True, False])




```python
((filedata > 50) & (filedata < 100))
# All values 50 < x < 100
```




    array([[False, False, False, False, False, False,  True, False],
           [False, False, False, False, False, False,  True, False]])




```python
(~((filedata > 50) & (filedata < 100)))
# Inverse of previous
```




    array([[ True,  True,  True,  True,  True,  True, False,  True],
           [ True,  True,  True,  True,  True,  True, False,  True]])



##### Indexing with a list


```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
a[[1, 2, 8]]
```




    array([2, 3, 9])



# Quizzes


```python
# quiz = np.array([[1, 2, 3, 4, 5],
#                   [6, 7, 8, 9, 10],
#                   [11, 12, 13, 14, 15],
#                   [16, 17, 18, 19, 20],
#                   [21, 22, 23, 24, 25],
#                   [26, 27, 38, 29, 30]])
quiz = np.arange(1, 31).reshape(6,5) # Easier method
print(quiz, '\n')
# Index 11, 12, 16, 17
print(quiz[2:4, 0:2], '\n')

# Index 2, 8, 14, 20 diagonally
print(quiz[ [0, 1, 2, 3], [1, 2, 3, 4]], '\n')
# Row 0,1,2,3 and col 1,2,3,4

# Index 4,5,24,25,29,30
print(quiz[ [0, 4, 5], 3:], '\n')
```

    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]
     [16 17 18 19 20]
     [21 22 23 24 25]
     [26 27 28 29 30]] 
    
    [[11 12]
     [16 17]] 
    
    [ 2  8 14 20] 
    
    [[ 4  5]
     [24 25]
     [29 30]] 
    
    
