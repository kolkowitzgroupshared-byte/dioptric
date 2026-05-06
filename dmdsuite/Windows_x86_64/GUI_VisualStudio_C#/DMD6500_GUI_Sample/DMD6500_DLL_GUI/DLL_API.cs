//BBS DLP6500 DLL C# wrapper


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

using System.Windows.Forms;


namespace DMD6500_DLL_API
{

    public struct Cmd
    {
        public const uint CMD_NOP = 0x10000000;
        public const uint CMD_OUTPUT = 0x80000000;
        public const uint CMD_GLOB_MIRRORCLOCKING = 0x40000000;
        public const uint CMD_GLOB_LOAD = 0xc0000000;
        public const uint CMD_GLOB_CLEAR = 0xe0000000;
        public const uint CMD_BLOCK_CLEAR = 0x30000000;
        public const uint CMD_FLOAT = 0x50000000;

        public const uint CMD_JUMP_TO = 0xf4000000;
        public const uint CMD_JUMP_RELATIVE = 0xf3000000;
        public const uint CMD_SEQ_END = 0xff000000;
        public const uint CMD_CALL = 0xf5000000;
        public const uint CMD_RETURN = 0xfe000000;

        public const uint CMD_WAIT_US_SINCE_MCP = 0xf8000000;
        public const uint CMD_WAIT_FOR_EVENT = 0xf9000000;
        public const uint CMD_CLEAR_EVENT = 0xba000000;

        public const uint CMD_IF_EVENT = 0xb8000000;
        public const uint CMD_IF_LATCHED_EVENT = 0xb9000000;
        public const uint CMD_IF_US_ELAPSED = 0xbf000000;  //typically use CMD_WAIT_US

        public const uint CMD_REGSET = 0xd0000000;

        public const uint CMD_IF_REG0_EQUALS_VALUE = 0xb0000000;
        public const uint CMD_IF_REG1_EQUALS_VALUE = 0xb1000000;
        public const uint CMD_IF_REG2_EQUALS_VALUE = 0xb2000000;
        public const uint CMD_IF_REG3_EQUALS_VALUE = 0xb3000000;
        public const uint CMD_IF_REG4_EQUALS_VALUE = 0xb4000000;
        public const uint CMD_IF_REG5_EQUALS_VALUE = 0xb5000000;
        public const uint CMD_IF_REG6_EQUALS_VALUE = 0xb6000000;
        public const uint CMD_IF_REG7_EQUALS_VALUE = 0xb7000000;

        public const uint CMD_IF_REG_EQUALS_REG = 0xbb000000;

        public const uint CMD_TIMERSTART = 0xa0000000;
        public const uint CMD_MISC_INTRPT = 0x90000000;
    }

    public struct RegSetMode
    {
        public const uint REG_SET_REG0 = 0x00000000;
        public const uint REG_SET_REG1 = 0x01000000;
        public const uint REG_SET_REG2 = 0x02000000;
        public const uint REG_SET_REG3 = 0x03000000;
        public const uint REG_SET_REG4 = 0x04000000;
        public const uint REG_SET_REG5 = 0x05000000;
        public const uint REG_SET_REG6 = 0x06000000;
        public const uint REG_SET_REG7 = 0x07000000;

        public const uint REG_INCR_REG0 = 0x08000000;
        public const uint REG_INCR_REG1 = 0x09000000;
        public const uint REG_INCR_REG2 = 0x0a000000;
        public const uint REG_INCR_REG3 = 0x0b000000;
        public const uint REG_INCR_REG4 = 0x0c000000;
        public const uint REG_INCR_REG5 = 0x0d000000;
        public const uint REG_INCR_REG6 = 0x0e000000;
        public const uint REG_INCR_REG7 = 0x0f000000;
    }

    public struct RegCompare
    {
        public const uint IF_COMP_A_REG0 = (0x0 << 11);
        public const uint IF_COMP_A_REG1 = (0x1 << 11);
        public const uint IF_COMP_A_REG2 = (0x2 << 11);
        public const uint IF_COMP_A_REG3 = (0x3 << 11);
        public const uint IF_COMP_A_REG4 = (0x4 << 11);
        public const uint IF_COMP_A_REG5 = (0x5 << 11);
        public const uint IF_COMP_A_REG6 = (0x6 << 11);
        public const uint IF_COMP_A_REG7 = (0x7 << 11);

        public const uint IF_COMP_B_REG0 = (0x0 << 8);
        public const uint IF_COMP_B_REG1 = (0x1 << 8);
        public const uint IF_COMP_B_REG2 = (0x2 << 8);
        public const uint IF_COMP_B_REG3 = (0x3 << 8);
        public const uint IF_COMP_B_REG4 = (0x4 << 8);
        public const uint IF_COMP_B_REG5 = (0x5 << 8);
        public const uint IF_COMP_B_REG6 = (0x6 << 8);
        public const uint IF_COMP_B_REG7 = (0x7 << 8);
    }


    public struct LoadMode
    {
        public const uint LOAD_FROM_LINENR = 0x07000000;

        public const uint LOAD_FROM_LINENR_IN_REG0 = 0x04000000;

        public const uint LOAD_PLANE_NR = 0x00000000;
        public const uint LOAD_REG0_PLUS_REG1 = 0x01000000;
        public const uint LOAD_REG0_PLUS_REG2 = 0x02000000;

        public const uint LOAD_OFFSET_REG0 = 0x08000000;
        public const uint LOAD_OFFSET_REG1 = 0x09000000;
        public const uint LOAD_OFFSET_REG2 = 0x0a000000;
        public const uint LOAD_OFFSET_REG3 = 0x0b000000;
        public const uint LOAD_OFFSET_REG4 = 0x0c000000;
        public const uint LOAD_OFFSET_REG5 = 0x0d000000;
        public const uint LOAD_OFFSET_REG6 = 0x0e000000;
        public const uint LOAD_OFFSET_REG7 = 0x0f000000;
    }

    public struct Events
    {
        public const uint EVENT_TRIG0_POSLEV = 0x00000001;
        public const uint EVENT_TRIG1_POSLEV = 0x00000002;
        public const uint EVENT_TRIG0_NEGLEV = 0x00000004;
        public const uint EVENT_TRIG1_NEGLEV = 0x00000008;
        public const uint EVENT_TRIG0_POSEDGE = 0x00000010;
        public const uint EVENT_TRIG1_POSEDGE = 0x00000020;
        public const uint EVENT_TRIG0_NEGEDGE = 0x00000040;
        public const uint EVENT_TRIG1_NEGEDGE = 0x00000080;
        public const uint EVENT_EVENT_SOFT = 0x00000100;
        public const uint EVENT_EVENT_HDMI = 0x00000200;
        public const uint EVENT_EVENT_USB = 0x00000400;
        public const uint EVENT_ALL = 0x0000ffff;//for clear event
    }

    public struct OutMode
    {
        public const uint OUT_PIN0 = 0x01000000;
        public const uint OUT_PIN1 = 0x02000000;
        public const uint OUT_DMD_FLAGS_PERM = 0x04000000;
        public const uint OUT_DMD_FLAGS_TEMP = 0x05000000;
        public const uint MODE_FLIP_X = (1 << 0);   //ModeReg and Sequence OUT_DMD_FLAGS_
        public const uint MODE_FLIP_Y = (1 << 1);   //ModeReg and Sequence OUT_DMD_FLAGS_
        public const uint MODE_COMPLEMENT = (1 << 2);   //ModeReg and Sequence OUT_DMD_FLAGS_
        public const uint MODE_TEMP_OVERRIDE = (1 << 7);   //Sequence OUT_DMD_FLAGS_TEMP only
    }



    public struct Cntrl
    {
        public const uint DMD_MODE_REG = 0;
        public const uint MODE_FLIP_X = (1 << 0);	//ModeReg and Sequence OUT_DMD_FLAGS_
        public const uint MODE_FLIP_Y = (1 << 1);	//ModeReg and Sequence OUT_DMD_FLAGS_
        public const uint MODE_COMPLEMENT = (1 << 2);	//ModeReg and Sequence OUT_DMD_FLAGS_
        public const uint MODE_TEMP_OVERRIDE = (1 << 7);	//Sequence OUT_DMD_FLAGS_TEMP only

        public const uint WDT_DISABLE = (1 << 16);	//ModeReg
        public const uint RESET2BLK_Z = (1 << 17);	//ModeReg

        public const uint SEQ_MODE_REG = 4;
        public const uint PAUSE = (1 << 0);
        public const uint FIFOCLEAR = (1 << 1);
        public const uint CANCELWAIT = (1 << 2);
        public const uint CLEAREVENTS = (1 << 4);
        public const uint CLEARTMPFLAGS = (1 << 5);

        public const uint COND_SOFT1 = (1 << 8);
        public const uint COND_SOFT2 = (1 << 9);
    }



    public struct Numbers
    {
        public const uint SEQ_BUFFER_SIZE = 131072;
        public const uint IDLE_SEQ_START = 131072;

        public const uint LOADWAITTIME = 100;

        public const uint REG_DECREMENT_PREFIX = 0x01000000;

        public const uint JUMP_FORWARD = 0;
        public const uint JUMP_BACKWARD = 0x00100000;
    }

    
    static class Api
    {
        //the DLL must be at the fixed path
        //const string dllpath = @"C:\test\DLP6500_DLL.dll";
        
        
        //pathless the DLL must be in the executable working directory
        const string dllpath = @"DLP6500_DLL.dll";





        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern void CalcAllPlanes(Byte[] greydata, Byte[,] array);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern void CalcPlane(Byte[] greydata, Byte[] plane, uint bitlevel);


        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern void SplitRGBPixels(IntPtr rgbValues, Byte[] rValues, Byte[] gValues, Byte[] bValues);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern void SplitRGBPixels(uint[] rgbValues, Byte[] rValues, Byte[] gValues, Byte[] bValues);




        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern IntPtr GetDevice();

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern void DeleteDevice(IntPtr devHandle);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int ListControllers(ref uint count);


        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int GetDevID(uint index, ref int ID);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int GetSerialNumber(int ID, sbyte[] serialnumber);



        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int RunSequence(IntPtr devHandle, uint startpos);


        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int StopSequence(IntPtr devHandle);



        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int Connect(IntPtr devHandle, int dev);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int Disconnect(IntPtr devHandle);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int IsConnected(IntPtr devHandle, ref bool connected);



        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int SendPlane(IntPtr devHandle, uint planenr, Byte[] PlaneBuffer);


        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int SendImageMono(IntPtr devHandle, uint planenr, Byte[] PixBuffer);


        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int SendImageRGB(IntPtr devHandle, uint planenr, uint[] ImgBuffer);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int SendImageRGB(IntPtr devHandle, uint planenr, IntPtr ImgBuffer);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int SetModeRegister(IntPtr devHandle, uint regnr, uint value);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int LoadPlaneToDLP(IntPtr devHandle, uint planenr);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int DLP_GlobalMCP(IntPtr devHandle);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int WriteCommand(IntPtr devHandle, uint seq_cmd);


        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int SendSequenceData(IntPtr devHandle, uint[] buffer, uint bufferoffset, uint size, uint startpos);


        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int StoreLogoToFlash(IntPtr devHandle, uint planenr);

        [DllImport(dllpath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
        public static extern int GetMaxPlaneNr(IntPtr Proj, ref uint maxplanenr);

    }


    class ApiWrapper
    {
        private IntPtr hdev = IntPtr.Zero;
        private string sernum = "";
        private int ID = 0 ;

        private int errorcount = 0;

        private int retval = -1;

        sbyte[] sn = new sbyte[16]; //ASCII serial number from USB chip



        public int devID { get { return ID; } }

        public string SerialNumber { get { return sernum; } }

        public IntPtr devhandle { get { return hdev; } }

        private int resulthandler
        {
            set
            {
                retval = value;

                if (value != 0)
                {
                    errorcount++;
                    //Throw an Exeption
                }
            }
        }

        public uint maxplanenr
        {
            get
            {
                uint logoplanenr = 0;

                resulthandler = Api.GetMaxPlaneNr(hdev, ref logoplanenr);

                return logoplanenr;
            }
        }

        public int callresult { get { return retval; } }


        public ApiWrapper()
        {
            hdev = Api.GetDevice();
        }

        ~ApiWrapper()
        {
            Api.DeleteDevice(hdev);
        }


        public int RunSequence(uint startpos)
        {
            resulthandler = Api.RunSequence(hdev, startpos);

            return retval;
        }

        public int AbortSequence()
        {
            resulthandler = Api.StopSequence(hdev);

            return retval;
        }

        public int PauseSequence(bool pause)
        {
            if (pause) resulthandler = Api.SetModeRegister(hdev, 1, 1);

            else resulthandler = Api.SetModeRegister(hdev, 1, 0);

            return retval;
        }

        public int ForceIdleSequence()
        {
            resulthandler = Api.StopSequence(hdev);

            resulthandler = Api.RunSequence(hdev, maxplanenr);

            return retval;
        }

        public int WriteCommand(uint seq_cmd)
        {
            resulthandler = Api.WriteCommand(hdev, seq_cmd);

            return retval;
        }

        public int SendSequenceData(uint[] buffer, uint bufferoffset, uint size, uint startpos)
        {
            resulthandler = Api.SendSequenceData(hdev, buffer, bufferoffset, size, startpos);

            return retval;
        }

        public int DLP_GlobalMCP()
        {
            resulthandler = WriteCommand(Cmd.CMD_GLOB_MIRRORCLOCKING);

            return retval;
        }

        public int LoadPlaneToDLP(uint planenr)
        {
            resulthandler = WriteCommand(Cmd.CMD_GLOB_LOAD | LoadMode.LOAD_PLANE_NR | planenr);

            return retval;
        }


        public int Connect()   //if only 1 Controller is connected to the PC
        {
            uint ncount = 0;

            resulthandler = Api.ListControllers(ref ncount);

            if (ncount == 1)
            {

                resulthandler = Api.GetDevID(0, ref ID);    //get the ID of the one and only Controller

                resulthandler = Api.Connect(hdev, ID);



                char[] usn = new char[16];  //Unicode 16 bit

                resulthandler =  Api.GetSerialNumber(ID, sn);


                for (int i = 0; i < 16; i++)
                {
                    usn[i] = (char)sn[i];   //ASCII to Unicode
                }

                sernum = new String(usn);   //String from Unicode char[]

                //MessageBox.Show("Connected with Controller having \nSerial Number " + sernum);
            }
            else
            {
                //MessageBox.Show(ncount + " Controllers found\n");

                resulthandler = -4;
            }

            return retval;
        }


        int Disconnect()
        {
            resulthandler = Api.Disconnect(hdev);

            return retval;
        }
        


    }




}
