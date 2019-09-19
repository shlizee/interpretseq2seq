# temporal_segmentation

## CMU MOCAP DATABASE
w: walking
r: running
j: jumping
d: directing traffic
s: soccer
b: basketball
w: wash window
bs: basketball signal

### Test Process
1. Three actions
  1. Simple, non-repeated 
     * w1,r1,j1 **checked** 
  2. Simple, repeated
     * w1,r1,j1,w2,r2 **checked** (step=8000,train_loss=0.0094,test_loss>0.05,error>1.05)
     * w1,r1,j1,w2,w3 **checked** (step=8000,train_loss=0.0057,test_loss>0.06,error>0.9)
     * w1,r1,j1,w2,r2,j3 **checked** (step=10000, train_loss=0.0079,test_loss>0.06,error>0.9,total_frames=2144)
     
     
2. Five actions
  1. Simple, non-repeated
    * w1,r1,j1,s1,b1 **checked** (step=5000,train_loss=0.0050,test_loss>0.06, error>0.9, total_frames=2452)
  2. Medieum, non-repeated
    * w1,r1,d1,s1,b1 **checked** (step=10000,batch_size=32,train_loss=0.0851,test_loss>0.8,error>1.2,total_frames=5065)
  
