
# coding: utf-8

# The Cell below imports the DLL and defines the functions needed for operation
#
# Modify the path to the DLL so that it is found on your PC

# In[1]:


import ctypes

# Specify  DLL  to load.

# DlpDLL = ctypes.WinDLL ("C:\\Users\\Edi\\Desktop\\DEV_DMD7000_USB\\DLP6500_DLL\\x64\\Release\\DLP6500_DLL.dll")
DlpDLL = ctypes.WinDLL ("C:\\Users\\jkdol\\OneDrive\\Documents\\Github\\dioptric\\dmdsuite\\Windows_x86_64\\DLL_x64\\x64\\Release\\DLP6500_DLL.dll")


# from  "C#" IntPtr hdev = GetDevice();

# Set up prototype and parameters for the desired function call.

GetDevice_Proto = ctypes.WINFUNCTYPE (ctypes.c_void_p)      # Return type is the first in ().


# Create the function.

GetDevice = GetDevice_Proto (("GetDevice", DlpDLL), ())



# from  "C#"  int Connect(IntPtr Proj, uint dev);

Connect_Proto = ctypes.WINFUNCTYPE (ctypes.c_int, ctypes.c_void_p, ctypes.c_int )

Connect_Params = (1, "p1", 0), (1, "p2", 0),

Connect = Connect_Proto(("Connect", DlpDLL), Connect_Params)


# from  "C#" int StopSequence(IntPtr Proj);

StopSequence_Proto = ctypes.WINFUNCTYPE (ctypes.c_int, ctypes.c_void_p)

StopSequence_Params = (1, "p1", 0),

StopSequence = StopSequence_Proto (("StopSequence", DlpDLL), StopSequence_Params)


# from  "C#"   int RunSequence(IntPtr Proj, uint startpos);

RunSequence_Proto = ctypes.WINFUNCTYPE (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)

RunSequence_Params = (1, "p1", 0), (1, "p2", 0),

RunSequence = RunSequence_Proto (("RunSequence", DlpDLL), RunSequence_Params)


# from  "C#"  int SendImageMono(IntPtr Proj, uint planenr, Byte[] PixBuffer);

SendImageMono_Proto = ctypes.WINFUNCTYPE (ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_ubyte*(1920*1080))

SendImageMono_Params = (1, "p1", 0), (1, "p2", 0), (1, "p3", 0),

SendImageMono_c = SendImageMono_Proto (("SendImageMono", DlpDLL), SendImageMono_Params)

def SendImageMono(hdev,  planenr, greyvals):
    return SendImageMono_c(hdev, planenr, (ctypes.c_ubyte * (1920*1080)).from_buffer( greyvals ))


from enum import IntFlag

class Numbers(IntFlag):
    IDLE_SEQ_START = 131072
    LOGOPLANE = 7705


# In[2]:


# do this only once per session!!!

hdev = GetDevice()      #Call the Function and get a handle to the device

Connect(hdev, 0)


# In[12]:


StopSequence(hdev)


# In[13]:


RunSequence(hdev, Numbers.IDLE_SEQ_START)


# In[5]:


from PIL import Image
#open a 1920 x 1080 RGB image
# im = Image.open("C:\\Users\\Edi\\Documents\\Testbilder\\VSMPTE133b_1920x1080.png")
im = Image.open("C:\\Users\\jkdol\\OneDrive\\Documents\\Github\\dioptric\\dmdsuite\\Windows_x86_64\\opticaltweezers.jpg")


# In[6]:


im.show()
print(im.getbands()) #should output 'R', 'G', 'B'

greyvals = bytearray(im.getdata(0))  # get the R channel as monochrome bytearray


# In[7]:


#send the grayscale image as bitplanes into planes 0 .. 7
#SendImageMono_c(hdev, 0, (ctypes.c_ubyte * (1920*1080)).from_buffer( greyvals ))
SendImageMono(hdev, 0, greyvals )


# In[8]:


StopSequence(hdev)
RunSequence(hdev, 0)


# In[9]:


# 'R', 'G', 'B'  24 planes

for color in range (len(im.getbands())):
    SendImageMono(hdev, color*8, bytearray(im.getdata(color)) )


# In[10]:


for i in range(125):
    SendImageMono(hdev, i*8, greyvals )

