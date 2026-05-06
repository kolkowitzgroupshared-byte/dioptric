

using System;
//using System.Collections.Generic;
//using System.ComponentModel;
//using System.Data;
using System.Drawing;
//using System.Linq;
//using System.Text;
using System.Threading;
using System.Windows.Forms;



//include the DLL Wrapper!!!
using DMD6500_DLL_API;



namespace DMD6500_DLL_GUI
{
    
    public partial class Form1 : Form
    {
                 
        IntPtr hdev = IntPtr.Zero;

        ApiWrapper APW = new ApiWrapper();

        

        
        public Form1()
        {
            InitializeComponent();


            hdev = APW.devhandle;

            int result = APW.Connect();

            if (result != 0) MessageBox.Show("Not connected to Controller\n");
            
        }


        private Bitmap tempimg;
        private System.Drawing.Imaging.BitmapData bmpData;

        
        //Helper Functions to get the Bitmap Data

        IntPtr GetBitmapPointer(Image bm)
        {
            if (bm == null) return  IntPtr.Zero;

            tempimg = new Bitmap(bm, 1920, 1080);

            if (tempimg == null) return IntPtr.Zero;

            Rectangle rect = new Rectangle(0, 0, 1920, 1080);

            bmpData = tempimg.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, tempimg.PixelFormat);

            return bmpData.Scan0;
        }

        void ReleaseBitmapPointer()
        {
            tempimg.UnlockBits(bmpData);
        }



        private void Stop_Click(object sender, EventArgs e)
        {
            Api.StopSequence(hdev);
            
            //RunSequence(hdev, 0);

            return;
            
          
        }

        private void PlaneSendMCP_Click(object sender, EventArgs e)
        {
            IntPtr pImg = GetBitmapPointer(pictureBox1.Image);

            if (pImg.Equals(IntPtr.Zero)) return;

            
            byte[] grayvals = new byte[1920 * 1080];

            byte[] plane = new byte[2048 * 1080 / 8];


            Api.SplitRGBPixels(pImg, grayvals, grayvals, grayvals);



            Api.StopSequence(hdev);

            uint planenr = (uint)numericUpDown_PlaneNr.Value;

            uint bitlev = (uint)numericUpDown_BitLev.Value;

            Api.CalcPlane(grayvals, plane, bitlev);


            Api.SendPlane(hdev, planenr, plane);




            Api.LoadPlaneToDLP(hdev, planenr);

            Api.DLP_GlobalMCP(hdev);



            ReleaseBitmapPointer();

        }

        private void Load_Click(object sender, EventArgs e)
        {
            var OfDlg = new OpenFileDialog();

            OfDlg.Filter = "Image Files(*.BMP;*.JPG;*.GIF;*.PNG;*.TIF)|*.BMP;*.JPG;*.GIF;*.PNG;*.TIF|All files (*.*)|*.*";

            if (OfDlg.ShowDialog(this) == DialogResult.OK)
            {
                pictureBox1.Load(OfDlg.FileName);
            }
        }

        private void Send_Click(object sender, EventArgs e)
        {
            IntPtr pImg = GetBitmapPointer(pictureBox1.Image);

            if ( pImg.Equals(IntPtr.Zero) ) return;


            byte[] grayvals = new byte[1920*1080];

            byte[] planebuf = new byte[2048/8 * 1080];

            //SplitRGBPixels(pImg, grayvals, grayvals, grayvals);

            //SendImageMono(hdev, 0, grayvals);

            uint planenr = (uint)numericUpDown_PlaneNr.Value;

            Api.SendImageRGB(hdev, planenr, pImg);


            ReleaseBitmapPointer();
        }

        private void IdleSequence_Click(object sender, EventArgs e)
        {
            Api.StopSequence(hdev);

            Api.RunSequence(hdev, Numbers.IDLE_SEQ_START);

        }

        private void Pause_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < 1000; i++)
            {
                Api.SetModeRegister(hdev, 1, 1);

                Api.SetModeRegister(hdev, 1, 0);

            }
        }

        private void MCP_Click(object sender, EventArgs e)
        {
            uint mode = 0;

            if(checkBox_X.Checked) mode |= Cntrl.MODE_FLIP_X;
            if(checkBox_Y.Checked) mode |= Cntrl.MODE_FLIP_Y;

            Api.SetModeRegister(hdev, Cntrl.DMD_MODE_REG, mode);

        }

        private void SendGray_Click(object sender, EventArgs e)
        {
            IntPtr pImg = GetBitmapPointer(pictureBox1.Image);

            if (pImg.Equals(IntPtr.Zero)) return;


            byte[] grayvals = new byte[1920 * 1080];

            byte[] planebuf = new byte[2048 / 8 * 1080];

            Api.SplitRGBPixels(pImg, grayvals, grayvals, grayvals);

            uint planenr = (uint)numericUpDown_PlaneNr.Value;

            int retval = Api.SendImageMono(hdev, planenr, grayvals);


            ReleaseBitmapPointer();

            //MessageBox.Show(retval.ToString());
        }
        


        private void LoadPlane_Click(object sender, EventArgs e)
        {
            uint planenr = (uint)numericUpDown_PlaneNr.Value;

            Api.LoadPlaneToDLP(hdev, planenr);

            Api.DLP_GlobalMCP(hdev);
        }

        private void GraySeq_Click(object sender, EventArgs e)
        {
            Api.StopSequence(hdev);
            
            Api.RunSequence(hdev, 0);
        }

        private void LoadTest_Click(object sender, EventArgs e)
        {
            MakeSequence();

            return;


            
            uint kstart = SetupStartupSequence();

            Api.SendSequenceData(hdev, SeqBufferPtr, 1000, 100, 1000);

            Api.RunSequence(hdev, kstart);

            
            
                        
        }




        uint[] SeqBufferPtr = new uint[Numbers.SEQ_BUFFER_SIZE];



        uint SetupStartupSequence()
        {
            uint k = 0;

            uint kstart = k;

            if (true)	//Idle Sequence
            {
                //k = Numbers.SEQ_BUFFER_SIZE;	//reserved after user sequence buffer

                //uint kstart = k;

                SeqBufferPtr[k++] = Cmd.CMD_OUTPUT | OutMode.OUT_DMD_FLAGS_TEMP | OutMode.MODE_TEMP_OVERRIDE | OutMode.MODE_COMPLEMENT;

                SeqBufferPtr[k++] = Cmd.CMD_GLOB_LOAD | LoadMode.LOAD_PLANE_NR | 5;

                SeqBufferPtr[k++] = Cmd.CMD_GLOB_MIRRORCLOCKING;

                SeqBufferPtr[k++] = Cmd.CMD_WAIT_US_SINCE_MCP | 200000;

                SeqBufferPtr[k++] = Cmd.CMD_OUTPUT | OutMode.OUT_DMD_FLAGS_TEMP | OutMode.MODE_TEMP_OVERRIDE | 0;

                SeqBufferPtr[k++] = Cmd.CMD_GLOB_LOAD | LoadMode.LOAD_PLANE_NR | 5;

                SeqBufferPtr[k++] = Cmd.CMD_GLOB_MIRRORCLOCKING;

                SeqBufferPtr[k++] = Cmd.CMD_WAIT_US_SINCE_MCP | 200000;

                SeqBufferPtr[k++] = Cmd.CMD_JUMP_TO | kstart;
            }

            return kstart;
        }

        
        private void MakeSequence()
        {
		    uint  k=1000;
		
		    uint  kstart = k;

		//for(uint i=0; i< 24 ; i++)
		//{

            SeqBufferPtr[k++] = Cmd.CMD_TIMERSTART;

            //SeqBufferPtr[k++] = Cmd.CMD_WAIT | 10000;

			SeqBufferPtr[k++] = Cmd.CMD_GLOB_LOAD |  LoadMode.LOAD_PLANE_NR| 5;

            
          

            SeqBufferPtr[k++] = Cmd.CMD_NOP;

            SeqBufferPtr[k++] = Cmd.CMD_WAIT_US_SINCE_MCP | 20000;

            //SeqBufferPtr[k++] = Cmd.CMD_OUTPUT;
            //SeqBufferPtr[k++] = Cmd.CMD_OUTPUT | OutMode.OUT_DMD_FLAGS_TEMP | OutMode.MODE_TEMP_OVERRIDE | OutMode.MODE_COMPLEMENT;

			SeqBufferPtr[k++] = Cmd.CMD_GLOB_MIRRORCLOCKING;

            SeqBufferPtr[k++] = Cmd.CMD_NOP;


            SeqBufferPtr[k++] = Cmd.CMD_WAIT_US_SINCE_MCP | 1000000;

            SeqBufferPtr[k++] = Cmd.CMD_GLOB_LOAD | LoadMode.LOAD_PLANE_NR | 7;

            //SeqBufferPtr[k++] = Cmd.CMD_WAIT | 10000;

            SeqBufferPtr[k++] = Cmd.CMD_NOP;
            SeqBufferPtr[k++] = Cmd.CMD_NOP;



            SeqBufferPtr[k++] = Cmd.CMD_NOP;

            SeqBufferPtr[k++] = Cmd.CMD_WAIT_US_SINCE_MCP | 20000;

            //SeqBufferPtr[k++] = Cmd.CMD_OUTPUT;
            //SeqBufferPtr[k++] = Cmd.CMD_OUTPUT | OutMode.OUT_DMD_FLAGS_TEMP | OutMode.MODE_TEMP_OVERRIDE | OutMode.MODE_COMPLEMENT;

            SeqBufferPtr[k++] = Cmd.CMD_GLOB_MIRRORCLOCKING;



            SeqBufferPtr[k++] = Cmd.CMD_NOP;

            SeqBufferPtr[k++] = (Cmd.CMD_WAIT_US_SINCE_MCP | 200000);
		//}

            SeqBufferPtr[k++] = (Cmd.CMD_JUMP_RELATIVE | (Numbers.JUMP_BACKWARD - 14));



            Api.SendSequenceData(hdev, SeqBufferPtr, kstart, k-kstart, kstart);

            Api.RunSequence(hdev, kstart);
        }

        private void StoreLogo_Click(object sender, EventArgs e)
        {
            IntPtr pImg = GetBitmapPointer(pictureBox1.Image);

            if (pImg.Equals(IntPtr.Zero)) return;


            byte[] grayvals = new byte[1920 * 1080];

            byte[] plane = new byte[2048 * 1080 / 8];


            Api.SplitRGBPixels(pImg, grayvals, grayvals, grayvals);



            Api.StopSequence(hdev);

            //uint planenr = (uint)numericUpDown_PlaneNr.Value;
            uint planenr = APW.maxplanenr;

            uint bitlev = (uint)numericUpDown_BitLev.Value;

            Api.CalcPlane(grayvals, plane, bitlev);


            Api.SendPlane(hdev, planenr, plane);




            Api.LoadPlaneToDLP(hdev, planenr);

            Api.DLP_GlobalMCP(hdev);



            ReleaseBitmapPointer();

            Api.StoreLogoToFlash(hdev, planenr);
        }

        private void ShowSernum_Click(object sender, EventArgs e)
        {
            uint ncont =0;

            Api.ListControllers(ref ncont);
            
            //MessageBox.Show(ncont + " Controllers found");

            //uint logoplanenr = 0;

            //Api.GetMaxPlaneNr(hdev, ref logoplanenr);

            //MessageBox.Show("LogoPlane Nr = " + APW.maxplanenr);



            MessageBox.Show("Serial Number = " + APW.SerialNumber);
        }
	}


    
}
