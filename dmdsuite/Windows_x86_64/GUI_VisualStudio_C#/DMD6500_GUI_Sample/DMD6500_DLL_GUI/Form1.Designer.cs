namespace DMD6500_DLL_GUI
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.button3 = new System.Windows.Forms.Button();
            this.button4 = new System.Windows.Forms.Button();
            this.button5 = new System.Windows.Forms.Button();
            this.button6 = new System.Windows.Forms.Button();
            this.button7 = new System.Windows.Forms.Button();
            this.button8 = new System.Windows.Forms.Button();
            this.button10 = new System.Windows.Forms.Button();
            this.button11 = new System.Windows.Forms.Button();
            this.button12 = new System.Windows.Forms.Button();
            this.button14 = new System.Windows.Forms.Button();
            this.button15 = new System.Windows.Forms.Button();
            this.numericUpDown_PlaneNr = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.numericUpDown_BitLev = new System.Windows.Forms.NumericUpDown();
            this.label2 = new System.Windows.Forms.Label();
            this.checkBox_X = new System.Windows.Forms.CheckBox();
            this.checkBox_Y = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown_PlaneNr)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown_BitLev)).BeginInit();
            this.SuspendLayout();
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(12, 325);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(105, 23);
            this.button1.TabIndex = 0;
            this.button1.Text = "Stop Sequence";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.Stop_Click);
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(12, 412);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(105, 23);
            this.button2.TabIndex = 1;
            this.button2.Text = "Send and Show Plane";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.PlaneSendMCP_Click);
            // 
            // pictureBox1
            // 
            this.pictureBox1.BackColor = System.Drawing.SystemColors.ControlDark;
            this.pictureBox1.Location = new System.Drawing.Point(12, 51);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(291, 166);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox1.TabIndex = 2;
            this.pictureBox1.TabStop = false;
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(12, 22);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(291, 23);
            this.button3.TabIndex = 3;
            this.button3.Text = "Load Image File";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.Load_Click);
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(12, 246);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(105, 23);
            this.button4.TabIndex = 4;
            this.button4.Text = "SendImageRGB";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.Send_Click);
            // 
            // button5
            // 
            this.button5.Location = new System.Drawing.Point(198, 285);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(105, 23);
            this.button5.TabIndex = 5;
            this.button5.Text = "Idle Sequence";
            this.button5.UseVisualStyleBackColor = true;
            this.button5.Click += new System.EventHandler(this.IdleSequence_Click);
            // 
            // button6
            // 
            this.button6.Location = new System.Drawing.Point(539, 377);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(105, 23);
            this.button6.TabIndex = 6;
            this.button6.Text = "PauseTest";
            this.button6.UseVisualStyleBackColor = true;
            this.button6.Click += new System.EventHandler(this.Pause_Click);
            // 
            // button7
            // 
            this.button7.Location = new System.Drawing.Point(163, 518);
            this.button7.Name = "button7";
            this.button7.Size = new System.Drawing.Size(140, 23);
            this.button7.TabIndex = 7;
            this.button7.Text = "Set Flip Mode";
            this.button7.UseVisualStyleBackColor = true;
            this.button7.Click += new System.EventHandler(this.MCP_Click);
            // 
            // button8
            // 
            this.button8.Location = new System.Drawing.Point(12, 285);
            this.button8.Name = "button8";
            this.button8.Size = new System.Drawing.Size(105, 23);
            this.button8.TabIndex = 8;
            this.button8.Text = "Send Image Gray";
            this.button8.UseVisualStyleBackColor = true;
            this.button8.Click += new System.EventHandler(this.SendGray_Click);
            // 
            // button10
            // 
            this.button10.Location = new System.Drawing.Point(12, 383);
            this.button10.Name = "button10";
            this.button10.Size = new System.Drawing.Size(105, 23);
            this.button10.TabIndex = 10;
            this.button10.Text = "Show Plane";
            this.button10.UseVisualStyleBackColor = true;
            this.button10.Click += new System.EventHandler(this.LoadPlane_Click);
            // 
            // button11
            // 
            this.button11.Location = new System.Drawing.Point(198, 325);
            this.button11.Name = "button11";
            this.button11.Size = new System.Drawing.Size(105, 23);
            this.button11.TabIndex = 11;
            this.button11.Text = "8 Bit Gray Seq";
            this.button11.UseVisualStyleBackColor = true;
            this.button11.Click += new System.EventHandler(this.GraySeq_Click);
            // 
            // button12
            // 
            this.button12.Location = new System.Drawing.Point(539, 348);
            this.button12.Name = "button12";
            this.button12.Size = new System.Drawing.Size(105, 23);
            this.button12.TabIndex = 12;
            this.button12.Text = "Load Test";
            this.button12.UseVisualStyleBackColor = true;
            this.button12.Click += new System.EventHandler(this.LoadTest_Click);
            // 
            // button14
            // 
            this.button14.Location = new System.Drawing.Point(163, 472);
            this.button14.Name = "button14";
            this.button14.Size = new System.Drawing.Size(140, 23);
            this.button14.TabIndex = 14;
            this.button14.Text = "Store as Logo";
            this.button14.UseVisualStyleBackColor = true;
            this.button14.Click += new System.EventHandler(this.StoreLogo_Click);
            // 
            // button15
            // 
            this.button15.Location = new System.Drawing.Point(12, 472);
            this.button15.Name = "button15";
            this.button15.Size = new System.Drawing.Size(133, 23);
            this.button15.TabIndex = 15;
            this.button15.Text = "Show Serial Number";
            this.button15.UseVisualStyleBackColor = true;
            this.button15.Click += new System.EventHandler(this.ShowSernum_Click);
            // 
            // numericUpDown_PlaneNr
            // 
            this.numericUpDown_PlaneNr.Location = new System.Drawing.Point(198, 249);
            this.numericUpDown_PlaneNr.Maximum = new decimal(new int[] {
            7705,
            0,
            0,
            0});
            this.numericUpDown_PlaneNr.Name = "numericUpDown_PlaneNr";
            this.numericUpDown_PlaneNr.Size = new System.Drawing.Size(105, 20);
            this.numericUpDown_PlaneNr.TabIndex = 16;
            this.numericUpDown_PlaneNr.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(149, 253);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(51, 13);
            this.label1.TabIndex = 17;
            this.label1.Text = "Plane Nr.";
            // 
            // numericUpDown_BitLev
            // 
            this.numericUpDown_BitLev.Location = new System.Drawing.Point(198, 415);
            this.numericUpDown_BitLev.Maximum = new decimal(new int[] {
            7,
            0,
            0,
            0});
            this.numericUpDown_BitLev.Name = "numericUpDown_BitLev";
            this.numericUpDown_BitLev.Size = new System.Drawing.Size(105, 20);
            this.numericUpDown_BitLev.TabIndex = 18;
            this.numericUpDown_BitLev.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.numericUpDown_BitLev.Value = new decimal(new int[] {
            7,
            0,
            0,
            0});
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(149, 417);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(48, 13);
            this.label2.TabIndex = 19;
            this.label2.Text = "Bit Level";
            // 
            // checkBox_X
            // 
            this.checkBox_X.AutoSize = true;
            this.checkBox_X.Location = new System.Drawing.Point(12, 522);
            this.checkBox_X.Name = "checkBox_X";
            this.checkBox_X.Size = new System.Drawing.Size(52, 17);
            this.checkBox_X.TabIndex = 20;
            this.checkBox_X.Text = "Flip X";
            this.checkBox_X.UseVisualStyleBackColor = true;
            // 
            // checkBox_Y
            // 
            this.checkBox_Y.AutoSize = true;
            this.checkBox_Y.Location = new System.Drawing.Point(89, 522);
            this.checkBox_Y.Name = "checkBox_Y";
            this.checkBox_Y.Size = new System.Drawing.Size(52, 17);
            this.checkBox_Y.TabIndex = 21;
            this.checkBox_Y.Text = "Flip Y";
            this.checkBox_Y.UseVisualStyleBackColor = true;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(314, 550);
            this.Controls.Add(this.checkBox_Y);
            this.Controls.Add(this.checkBox_X);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.numericUpDown_BitLev);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.numericUpDown_PlaneNr);
            this.Controls.Add(this.button15);
            this.Controls.Add(this.button14);
            this.Controls.Add(this.button12);
            this.Controls.Add(this.button11);
            this.Controls.Add(this.button10);
            this.Controls.Add(this.button8);
            this.Controls.Add(this.button7);
            this.Controls.Add(this.button6);
            this.Controls.Add(this.button5);
            this.Controls.Add(this.button4);
            this.Controls.Add(this.button3);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            this.Name = "Form1";
            this.Text = "BBS DLP6500 ALC GUI";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown_PlaneNr)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown_BitLev)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.Button button6;
        private System.Windows.Forms.Button button7;
        private System.Windows.Forms.Button button8;
        private System.Windows.Forms.Button button10;
        private System.Windows.Forms.Button button11;
        private System.Windows.Forms.Button button12;
        private System.Windows.Forms.Button button14;
        private System.Windows.Forms.Button button15;
        private System.Windows.Forms.NumericUpDown numericUpDown_PlaneNr;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.NumericUpDown numericUpDown_BitLev;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.CheckBox checkBox_X;
        private System.Windows.Forms.CheckBox checkBox_Y;
    }
}

