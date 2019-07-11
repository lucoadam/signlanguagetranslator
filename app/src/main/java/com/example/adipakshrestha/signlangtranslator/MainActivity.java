package com.example.adipakshrestha.signlangtranslator;

import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.hardware.Camera;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.Toast;

import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.BOWImgDescriptorExtractor;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.KNearest;
import org.opencv.ml.NormalBayesClassifier;
import org.opencv.ml.SVM;
import org.opencv.ml.TrainData;
import org.opencv.objdetect.BaseCascadeClassifier;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.objdetect.Objdetect;
import org.opencv.video.BackgroundSubtractor;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    Button btnSnap;
    Bitmap imageBitmap;
    Bundle extras;
    ImageView imageView;
    Bitmap originalBitmap;
    SeekBar stopHue, stopSaturation, stopValue;
    SeekBar startHue, startSaturation, startValue;
    LinearLayout llH,llS,llV;
    LinearLayout sllH,sllS,sllV;

    static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final String IMAGE_DIRECTORY = "/FinalProjectData";
    public static final int JAVA_DETECTOR = 0;
    private int mJavaDetectorType = JAVA_DETECTOR;
    private int GALLERY = 1, CAMERA = 2;
    private final static String TAG = "CameraEnhance_ImageDFT";

    File wallpaperDirectory;
    File f;
    long lFileName;

    private Camera mCamera;
    private CameraPreview mPreview;
    Objdetect objdetect;

    CascadeClassifier faceCascade;
    CascadeClassifier handCascade;
    int absoluteFaceSize = 0, absoluteHandSize = 0;
    Scalar sMinValues, sMaxValues;
    String sCheck;
    int stopH, startH, stopS, startS, stopV, startV;
    Mat blurredImage;
    Mat hsvImage;
    Mat mask;
    Mat morphOutput;

    static {
        if(!OpenCVLoader.initDebug()){
            Log.i("openCv","Open cv loaded successfully");
        }else{
            Log.i("openCv","Open Cv cannot be loaded");
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.process_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        if(id == R.id.original){
            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            imageView.setImageBitmap(originalBitmap);
            return true;
        }

        if (id == R.id.grayScale) {
            //Intent intent1 = new Intent(this,MyActivity.class);
            //this.startActivity(intent1);
            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(imageBitmap,imgMat);
            Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);


            Utils.matToBitmap(imgMat,imageBitmap);
            //String path = saveImage(imageBitmap);
            imageView.setImageBitmap(imageBitmap);
            return true;
        }

        if (id == R.id.adaptiveThresholding) {
            //Toast.makeText(this, "Setting", Toast.LENGTH_LONG).show();
            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(imageBitmap,imgMat);
            Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);
            Imgproc.adaptiveThreshold(imgMat,imgMat,125,Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,11,12);
            Utils.matToBitmap(imgMat,imageBitmap);
            imageView.setImageBitmap(imageBitmap);
            return true;
        }

        if(id == R.id.erode){
            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(imageBitmap,imgMat);
            //Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);
            //Imgproc.adaptiveThreshold(imgMat,imgMat,125,Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,11,12);
            //Imgproc.Canny(imgMat,imgMat,60,60*3);
            // Creating an empty matrix to store the result

            Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));

            Imgproc.erode(mask, morphOutput, erodeElement);
            Imgproc.erode(mask, morphOutput, erodeElement);



            Utils.matToBitmap(morphOutput,imageBitmap);
            imageView.setImageBitmap(imageBitmap);
            return true;
        }

        if(id == R.id.dilate){
            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(imageBitmap,imgMat);
            //Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);
            //Imgproc.adaptiveThreshold(imgMat,imgMat,125,Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,11,12);
            //Imgproc.Canny(imgMat,imgMat,60,60*3);
            // Creating an empty matrix to store the result
            Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(24, 24));
            Imgproc.dilate(mask, morphOutput, dilateElement);
            Imgproc.dilate(mask, morphOutput, dilateElement);
            Utils.matToBitmap(morphOutput,imageBitmap);
            imageView.setImageBitmap(imageBitmap);
            return true;
        }

        if(id == R.id.boundaryDetection){
            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(imageBitmap,imgMat);
            Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);
            Imgproc.threshold(imgMat,imgMat,75,255,Imgproc.THRESH_BINARY);
            Utils.matToBitmap(imgMat,imageBitmap);
            imageView.setImageBitmap(imageBitmap);
            Toast.makeText(MainActivity.this,"Error",Toast.LENGTH_LONG);
            return  true;
        }

        if(id == R.id.fourierDes){
            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(),CvType.CV_8UC1);
            Utils.bitmapToMat(imageBitmap,imgMat);
            Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);

            int m = Core.getOptimalDFTSize(imgMat.rows());
            int n = Core.getOptimalDFTSize(imgMat.cols());
            Mat padded = new Mat(new Size(n,m),CvType.CV_64FC2);
            Core.copyMakeBorder(imgMat,padded,0,m - imgMat.rows(),0,n - imgMat.cols(),Core.BORDER_CONSTANT);
            List<Mat> planes = new ArrayList<Mat>();
            //imgMat1.convertTo(imgMat1,CvType.CV_32F);
            planes.add(padded);
            planes.add(Mat.zeros(padded.rows(),padded.cols(),CvType.CV_64FC2));
            for(Mat mat : planes) {
                Log.d(TAG, "Depth: " +mat.depth()+ " Size: " +mat.size().width +" X " +mat.size().height);
            }
            Mat complexI = new Mat(padded.rows(),padded.cols(),CvType.CV_64FC2);
            Mat complexI2 = Mat
                    .zeros(padded.rows(), padded.cols(), CvType.CV_64FC2);
            Core.merge(planes,complexI);
            Core.dft(complexI,complexI2);

            Core.split(complexI2,planes);
            Mat mag = new Mat(planes.get(0).size(), planes.get(0).type());
            Core.magnitude(planes.get(0),planes.get(1),mag);

            Mat magI = mag;
            Mat magI2 = new Mat(magI.size(), magI.type());
            Mat magI3 = new Mat(magI.size(), magI.type());
            Mat magI4 = new Mat(magI.size(), magI.type());
            Mat magI5 = new Mat(magI.size(), magI.type());
            Core.add(magI,Mat.ones(padded.rows(),padded.cols(),CvType.CV_64FC1),magI2);
            Core.log(magI2,magI3);
            Mat crop = new Mat(magI3, new Rect(0, 0, magI3.cols() & -2,
                    magI3.rows() & -2));

            magI4 = crop.clone();
            int cx = magI4.cols() / 2;
            int cy = magI4.rows() / 2;

            Rect q0Rect = new Rect(0, 0, cx, cy);
            Rect q1Rect = new Rect(cx, 0, cx, cy);
            Rect q2Rect = new Rect(0, cy, cx, cy);
            Rect q3Rect = new Rect(cx, cy, cx, cy);

            Mat q0 = new Mat(magI4, q0Rect); // Top-Left - Create a ROI per quadrant
            Mat q1 = new Mat(magI4, q1Rect); // Top-Right
            Mat q2 = new Mat(magI4, q2Rect); // Bottom-Left
            Mat q3 = new Mat(magI4, q3Rect); // Bottom-Right

            Mat tmp = new Mat(); // swap quadrants (Top-Left with Bottom-Right)
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);
            q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
            q2.copyTo(q1);
            tmp.copyTo(q2);
            Core.normalize(magI4, magI5, 0, 255, Core.NORM_MINMAX);
            Mat realResult = new Mat(magI5.size(), CvType.CV_8UC1);
            magI5.convertTo(realResult, CvType.CV_8UC1);

            Utils.matToBitmap(magI5,imageBitmap);
            imageView.setImageBitmap(imageBitmap);

            return true;
        }

        if(id == R.id.detectFace){

            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(),CvType.CV_8U);
            Mat imgMatOriginal = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(),CvType.CV_8U);
            Utils.bitmapToMat(imageBitmap,imgMat);
            Utils.bitmapToMat(imageBitmap,imgMatOriginal);
            Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);
            Imgproc.equalizeHist(imgMat,imgMat);

            String path = Environment.getExternalStorageDirectory()+"/faceData/haarcascade_frontalface_default.xml";
            Log.d("StringPath",path);
            faceCascade = new CascadeClassifier(path);
            if(this.absoluteFaceSize == 0){
                int height = imgMat.rows();
                if(Math.round(height*0.1f)>0){
                    this.absoluteFaceSize = Math.round(height*0.1f);
                }
            }

            MatOfRect faces = new MatOfRect();

            if(!faceCascade.empty()){
                Log.d("AfterFaceCascade","got face Cascade");
                faceCascade.detectMultiScale(imgMat,faces,1.1,2,2,new Size(absoluteFaceSize,absoluteFaceSize),new Size());
                Rect[] arrayFaces = faces.toArray();
                for(int i=0;i<arrayFaces.length;i++){
                    Imgproc.rectangle(imgMatOriginal,arrayFaces[i].tl(),arrayFaces[i].br(),new Scalar(0,255,0,255),3);
                }

                //faces.convertTo(imgMatOriginal,CvType.CV_8U);
                Utils.matToBitmap(imgMatOriginal,imageBitmap);
                imageView.setImageBitmap(imageBitmap);
            }else{
                Log.d("MainError","error in face cascade");
            }
            return  true;
        }

        if(id== R.id.detectHand){
            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(),CvType.CV_8U);
            Mat imgMatOriginal = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(),CvType.CV_8U);
            Utils.bitmapToMat(imageBitmap,imgMat);
            Utils.bitmapToMat(imageBitmap,imgMatOriginal);

            blurredImage = new Mat();
            hsvImage = new Mat();
            mask = new Mat();
            morphOutput = new Mat();

            //Remove Noise using blurring and converting from BGR to HSV
            Imgproc.blur(imgMat,blurredImage,new Size(7,7));
            Imgproc.cvtColor(blurredImage,hsvImage,Imgproc.COLOR_BGR2HSV);

            llH.setVisibility(View.VISIBLE);
            llS.setVisibility(View.VISIBLE);
            llV.setVisibility(View.VISIBLE);
            sllH.setVisibility(View.VISIBLE);
            sllS.setVisibility(View.VISIBLE);
            sllV.setVisibility(View.VISIBLE);

            startHue.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    startH = startHue.getProgress();
                    String st = String.valueOf(startH);
                    btnSnap.setText(st);
                    sMinValues = new Scalar(startH,startS,startV);
                    sMaxValues = new Scalar(stopH,stopS,stopV);
                    Core.inRange(hsvImage,sMinValues,sMaxValues,mask);
                    morphOutput = mask;
                    Utils.matToBitmap(mask,imageBitmap);
                    imageView.setImageBitmap(imageBitmap);
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {

                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {
                    btnSnap.setText("CAPTURE OR SELECTIMAGE");
                }
            });
            stopHue.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    stopH = stopHue.getProgress();
                    String st = String.valueOf(stopH);
                    btnSnap.setText(st);
                    sMinValues = new Scalar(startH,startS,startV);
                    sMaxValues = new Scalar(stopH,stopS,stopV);
                    Core.inRange(hsvImage,sMinValues,sMaxValues,mask);
                    morphOutput = mask;
                    Utils.matToBitmap(mask,imageBitmap);
                    imageView.setImageBitmap(imageBitmap);
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {

                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {
                    btnSnap.setText("CAPTURE OR SELECTIMAGE");
                }
            });
            startSaturation.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    startS = startSaturation.getProgress();
                    String st = String.valueOf(startS);
                    btnSnap.setText(st);
                    sMinValues = new Scalar(startH,startS,startV);
                    sMaxValues = new Scalar(stopH,stopS,stopV);
                    Core.inRange(hsvImage,sMinValues,sMaxValues,mask);
                    morphOutput = mask;
                    Utils.matToBitmap(mask,imageBitmap);
                    imageView.setImageBitmap(imageBitmap);
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {

                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {
                    btnSnap.setText("CAPTURE OR SELECTIMAGE");
                }
            });
            stopSaturation.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    stopS = stopSaturation.getProgress();
                    String st = String.valueOf(stopS);
                    btnSnap.setText(st);
                    sMinValues = new Scalar(startH,startS,startV);
                    sMaxValues = new Scalar(stopH,stopS,stopV);
                    Core.inRange(hsvImage,sMinValues,sMaxValues,mask);
                    morphOutput = mask;
                    Utils.matToBitmap(mask,imageBitmap);
                    imageView.setImageBitmap(imageBitmap);
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {

                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {

                    btnSnap.setText("CAPTURE OR SELECTIMAGE");
                }
            });
            startValue.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    startV = startValue.getProgress();
                    String st = String.valueOf(startV);
                    btnSnap.setText(st);
                    sMinValues = new Scalar(startH,startS,startV);
                    sMaxValues = new Scalar(stopH,stopS,stopV);
                    Core.inRange(hsvImage,sMinValues,sMaxValues,mask);
                    morphOutput = mask;
                    Utils.matToBitmap(mask,imageBitmap);
                    imageView.setImageBitmap(imageBitmap);
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {

                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {

                    btnSnap.setText("CAPTURE OR SELECTIMAGE");
                }
            });
            stopValue.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    stopV = stopValue.getProgress();
                    String st = String.valueOf(stopV);
                    btnSnap.setText(st);
                    sMinValues = new Scalar(startH,startS,startV);
                    sMaxValues = new Scalar(stopH,stopS,stopV);
                    Core.inRange(hsvImage,sMinValues,sMaxValues,mask);
                    morphOutput = mask;
                    Utils.matToBitmap(mask,imageBitmap);
                    imageView.setImageBitmap(imageBitmap);
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {

                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {

                    btnSnap.setText("CAPTURE OR SELECTIMAGE");
                }
            });


            /*Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);
            Imgproc.equalizeHist(imgMat,imgMat);

            String path = Environment.getExternalStorageDirectory()+"/faceData/Hand.Cascade.1.xml";
            Log.d("StringPath",path);
            handCascade = new CascadeClassifier(path);
            if(this.absoluteHandSize == 0){
                int height = imgMat.rows();
                if(Math.round(height*0.7f)>0){
                    this.absoluteHandSize = Math.round(height*0.7f);
                }
            }

            MatOfRect hands = new MatOfRect();

            if(!handCascade.empty()){
                Log.d("AfterFaceCascade","got face Cascade");
                handCascade.detectMultiScale(imgMat,hands,1.1,2,0 | Objdetect.CASCADE_SCALE_IMAGE,new Size(absoluteFaceSize,absoluteFaceSize),new Size());
                Rect[] arrayHand = hands.toArray();
                for(int i=0;i<arrayHand.length;i++){
                    Imgproc.rectangle(imgMatOriginal,arrayHand[i].tl(),arrayHand[i].br(),new Scalar(0,255,0,255),3);
                }

                //faces.convertTo(imgMatOriginal,CvType.CV_8U);
                Utils.matToBitmap(imgMatOriginal,imageBitmap);
                imageView.setImageBitmap(imageBitmap);
            }else{
                Log.d("MainError","error in face cascade");
            }*/
            return  true;
        }

        if(id==R.id.drawObject){
            try{
                File oFile = new File(Environment.getExternalStorageDirectory().getAbsolutePath()+IMAGE_DIRECTORY,String.valueOf(lFileName)+".jpg");
                FileInputStream fileInputStream = new FileInputStream(oFile);
                imageBitmap = BitmapFactory.decodeStream(fileInputStream);
                fileInputStream.close();
            }catch (FileNotFoundException e){
                e.printStackTrace();
            }catch (IOException e){
                e.printStackTrace();
            }
            Mat originalMat = new Mat();
            Utils.bitmapToMat(imageBitmap,originalMat);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat heirarchy = new Mat();
            //find contours
            Imgproc.findContours(morphOutput,contours,heirarchy,Imgproc.RETR_CCOMP,Imgproc.CHAIN_APPROX_SIMPLE);
            //if any contours exist
            if(heirarchy.size().height>0 && heirarchy.size().width>0){
                //for each contour display it in red
                for( int idx = 0; idx>=0; idx=(int) heirarchy.get(0,idx)[0]){
                    Imgproc.drawContours(originalMat,contours,idx,new Scalar(255,0,0),10);
                }
            }
            //Double contourArea = Imgproc.contourArea(originalMat);
            Utils.matToBitmap(originalMat,imageBitmap);
            imageView.setImageBitmap(imageBitmap);
            //btnSnap.setText(Double.toString(contourArea));

            return true;
        }

        if(id == R.id.blurr){
            imageBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(),CvType.CV_8UC1);
            Utils.bitmapToMat(imageBitmap,imgMat);
            //Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);
            Imgproc.GaussianBlur(imgMat,imgMat,new Size(45,45),0);
            Utils.matToBitmap(imgMat,imageBitmap);
            imageView.setImageBitmap(imageBitmap);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnSnap = findViewById(R.id.btnCapture);
        imageView = findViewById(R.id.imageView);
        stopHue = findViewById(R.id.seekBarH);
        stopSaturation = findViewById(R.id.seekBarS);
        stopValue = findViewById(R.id.seekBarV);
        startHue = findViewById(R.id.sSeekBarH);
        startSaturation = findViewById(R.id.sSeekBarS);
        startValue = findViewById(R.id.sSeekBarV);
        sllH = findViewById(R.id.sLinearH);
        sllS = findViewById(R.id.sLinearS);
        sllV = findViewById(R.id.sLinearV);
        llH = findViewById(R.id.linearH);
        llS = findViewById(R.id.linearS);
        llV = findViewById(R.id.linearV);

       // Create an instance of Camera
        mCamera = getCameraInstance();

        // Create our Preview view and set it as the content of our activity.
        /*mPreview = new CameraPreview(this, mCamera);
        FrameLayout preview = (FrameLayout) findViewById(R.id.cameraSurfaceView);
        preview.addView(mPreview);*/

        btnSnap.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //dispatchTakePictureIntent();
                llH.setVisibility(View.GONE);
                llS.setVisibility(View.GONE);
                llV.setVisibility(View.GONE);
                sllH.setVisibility(View.GONE);
                sllS.setVisibility(View.GONE);
                sllV.setVisibility(View.GONE);
                showPictureDialog();
            }
        });



    }

    /** A safe way to get an instance of the Camera object. */
    private Camera getCameraInstance(){
        Camera c = null;
        try {
            c = Camera.open(); // attempt to get a Camera instance
        }
        catch (Exception e){
            // Camera is not available (in use or does not exist)
            Toast.makeText(this,"Camera is not available in your device",Toast.LENGTH_SHORT);
        }
        return c; // returns null if camera is unavailable
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode,resultCode,data);


        if (resultCode == this.RESULT_CANCELED) {
            return;
        }
        if (requestCode == CAMERA && resultCode == RESULT_OK) {
            extras = data.getExtras();
            /*Intent intent = new Intent(this,ImageProcess.class);
            intent.putExtra("data",extras);
            startActivity(intent);*/
            imageBitmap = (Bitmap) extras.get("data");
            originalBitmap = imageBitmap;
            /*Mat imgMat = new Mat(imageBitmap.getWidth(),imageBitmap.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(imageBitmap,imgMat);
            Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);
            Imgproc.adaptiveThreshold(imgMat,imgMat,125,Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,11,12);
            //Imgproc.Canny(imgMat,imgMat,60,60*3);
            // Creating an empty matrix to store the result
            Mat dst = new Mat();

            // Preparing the kernel matrix object
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                    new Size((1*1) + 1, (1*1)+1));

            // Applying erode on the Image
            Imgproc.erode(imgMat, dst, kernel);
            Imgproc.erode(imgMat, dst, kernel);
            //Imgproc.dilate(imgMat,dst,kernel);
            Utils.matToBitmap(dst,imageBitmap);*/
            String path = saveImage(imageBitmap);
            Toast.makeText(MainActivity.this, "Image Saved!" , Toast.LENGTH_SHORT).show();
            imageView.setImageBitmap(imageBitmap);
        }else if (requestCode == GALLERY){
            if (data != null) {
                Uri contentURI = data.getData();
                try {
                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), contentURI);
                    /*Mat imgMat = new Mat(bitmap.getWidth(),bitmap.getHeight(), CvType.CV_8UC1);
                    Utils.bitmapToMat(bitmap,imgMat);
                    Imgproc.cvtColor(imgMat,imgMat,Imgproc.COLOR_RGB2GRAY);
                    Imgproc.adaptiveThreshold(imgMat,imgMat,125,Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,11,12);
                    //Imgproc.Canny(imgMat,imgMat,60,60*3);

                    // Creating an empty matrix to store the result
                    Mat dst = new Mat();

                    // Preparing the kernel matrix object
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                            new  Size((1*1) + 1, (1*1)+1));

                    // Applying erode on the Image
                    Imgproc.erode(imgMat, dst, kernel);
                    Imgproc.erode(imgMat, dst, kernel);
                    Imgproc.threshold(dst,dst,25,255,Imgproc.THRESH_BINARY_INV);

                    //Imgproc.Canny(dst,dst,60,60*3);
                    //Imgproc.dilate(imgMat,dst,kernel);
                    Utils.matToBitmap(dst,bitmap);*/
                    String path = saveImage(bitmap);
                    Toast.makeText(MainActivity.this, "Image Saved!", Toast.LENGTH_SHORT).show();
                    imageView.setImageBitmap(bitmap);

                } catch (IOException e) {
                    e.printStackTrace();
                    Toast.makeText(MainActivity.this, "Failed!", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    private void showPictureDialog(){
        AlertDialog.Builder pictureDialog = new AlertDialog.Builder(this);
        pictureDialog.setTitle("Select Action");
        String[] pictureDialogItems = {
                "Select photo from gallery",
                "Capture photo from camera" };
        pictureDialog.setItems(pictureDialogItems,
                new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        switch (which) {
                            case 0:
                                choosePhotoFromGallary();
                                break;
                            case 1:
                                takePhotoFromCamera();
                                break;
                        }
                    }
                });
        pictureDialog.show();
    }

    public void choosePhotoFromGallary() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

        startActivityForResult(galleryIntent, GALLERY);
    }

    private void takePhotoFromCamera() {
        Intent intent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, CAMERA);
    }

    public String saveImage(Bitmap myBitmap) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        myBitmap.compress(Bitmap.CompressFormat.JPEG, 90, bytes);
        wallpaperDirectory = new File(
                Environment.getExternalStorageDirectory() + IMAGE_DIRECTORY);
        // have the object build the directory structure, if needed.
        if (!wallpaperDirectory.exists()) {
            wallpaperDirectory.mkdirs();
        }

        try {
            lFileName = Calendar.getInstance().getTimeInMillis();
            f = new File(wallpaperDirectory, Calendar.getInstance()
                    .getTimeInMillis() + ".jpg");
            f.createNewFile();
            FileOutputStream fo = new FileOutputStream(f);
            fo.write(bytes.toByteArray());
            MediaScannerConnection.scanFile(this,
                    new String[]{f.getPath()},
                    new String[]{"image/jpeg"}, null);
            fo.close();
            Log.d("TAG", "File Saved::--->" + f.getAbsolutePath());

            return f.getAbsolutePath();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
        return "";
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }
}
