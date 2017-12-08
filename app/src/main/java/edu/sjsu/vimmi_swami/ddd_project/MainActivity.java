package edu.sjsu.vimmi_swami.ddd_project;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;

import android.location.Location;
import android.location.LocationManager;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.Settings;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.widget.Toast;
import android.util.Log;
import android.view.View;
import android.telephony.SmsManager;
import android.app.Activity;
import android.widget.Button;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.tasks.OnSuccessListener;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private static final int SEND_SMS_PERMISSION = 0;
    private SensorManager senSensorManager;
    private LocationManager manager;
    private Button start_stop;
    private int count = 0;
    static final int permissionCoarse = 0;
    static final int permissionFine = 0;
    static final int permissionSMS = 0;
    private Boolean appStarted = false;
    private Sensor senAccelerometer, senMagnetometer, senGyrometer;
    private float acc_x = 0, acc_y = 0, acc_z = 0, gyro_x = 0, gyro_y = 0, gyro_z = 0, mag_x = 0, mag_y = 0, mag_z = 0;
    private FusedLocationProviderClient mFusedLocationClient;
    private Context contextp;

    Location locationGPS = null;
    List<String> phoneNo;
    String message = "Hi!, I have been detected by the App DDD to be distracted driving!";

    public static String convertStreamToString(InputStream is) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        StringBuilder sb = new StringBuilder();
        String line = null;
        while ((line = reader.readLine()) != null) {
            sb.append(line);
        }
        reader.close();
        return sb.toString();
    }

    private void gpsCheck() {

        while (!manager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            Toast toast = Toast.makeText(getApplicationContext(), "Please enable your GPS!", Toast.LENGTH_SHORT);
            toast.setGravity(Gravity.CENTER, 0, 0);
            toast.show();
            Intent intent = new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS);
            startActivity(intent);
        }
    }
    private void missingPermissions(){
        Toast toast = Toast.makeText(getApplicationContext(), "Some permissions are missing, please enable them first, exiting app!", Toast.LENGTH_LONG);
        toast.setGravity(Gravity.CENTER, 0, 0);
        toast.show();


        Intent intent = new Intent();
        intent.setAction(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        Uri uri = Uri.fromParts("package", getPackageName(), null);
        intent.setData(uri);
        startActivity(intent);
        finish();
    }

    private void killActivity() {
        MainActivity.this.finish();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        if (getIntent().getExtras() != null && getIntent().getExtras().getBoolean("EXIT", false)) {
            Log.d("Oncreate", "intent sent exist");
            Toast toast = Toast.makeText(getApplicationContext(), "Exiting app!", Toast.LENGTH_SHORT);
            toast.setGravity(Gravity.CENTER, 0, 0);
            toast.show();
            killActivity();
        }else {
            manager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
            senSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
            senAccelerometer = senSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            senMagnetometer = senSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
            senGyrometer = senSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
            contextp = this;
            if (senAccelerometer == null || senMagnetometer == null || senGyrometer == null) {
                Toast toast = Toast.makeText(getApplicationContext(), "Some Sensors are not supported! The app will exit now!", Toast.LENGTH_LONG);
                toast.setGravity(Gravity.CENTER, 0, 0);
                toast.show();
                killActivity();
            } else if (!(CheckPermission(contextp, Manifest.permission.ACCESS_FINE_LOCATION) && CheckPermission(contextp, Manifest.permission.ACCESS_COARSE_LOCATION) && CheckPermission(contextp, Manifest.permission.SEND_SMS) && CheckPermission(contextp, Manifest.permission.READ_CONTACTS))) {
                Log.d("Permission", "MISING");
                missingPermissions();


            } else {
                // If contacts file exist then skip, else, start intent to ask and store emergency contacts
                Context mContext = getApplicationContext();
                File file = new File(mContext.getCacheDir(), "contacts.txt");
                if (!file.exists()) {
                    Intent i = new Intent(this, Contacts.class);
                    startActivity(i);
                } else
                {
                    try {
                            FileInputStream fin = new FileInputStream(file);
                            String ret = convertStreamToString(fin);
                            Log.d("CONTACTSLIST", ret);

                            //Make sure you close all streams.
                            fin.close();
                            List<String> temp = Arrays.asList(ret.split(","));
                            Log.d("CONTACT FETCH", temp.get(0).split(":")[1]+"*"+temp.get(1).split(":")[1]+"*"+temp.get(2).split(":")[1]);
                            phoneNo = new ArrayList<String>();
                            phoneNo.add(temp.get(0).split(":")[1]);
                            phoneNo.add(temp.get(1).split(":")[1]);
                            phoneNo.add(temp.get(2).split(":")[1]);
                        } catch (FileNotFoundException e) {
                        Log.e("Read Contacts",e.getMessage());
                    } catch (IOException e) {
                        Log.e("Read Contacts",e.getMessage());
                    } catch (Exception e) {
                        Log.e("Read Contacts",e.getMessage());
                    }

                }

                // Check for GPS and location and SMS permissions
                mFusedLocationClient = LocationServices.getFusedLocationProviderClient(this);
                gpsCheck();

                setContentView(R.layout.activity_main);
                Toolbar myToolbar = (Toolbar) findViewById(R.id.my_toolbar);
                setSupportActionBar(myToolbar);
                getSupportActionBar().setTitle("");

                //senSensorManager.registerListener(MainActivity.this, senAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
                //senSensorManager.registerListener(MainActivity.this, senMagnetometer, SensorManager.SENSOR_DELAY_NORMAL);
                //senSensorManager.registerListener(MainActivity.this, senGyrometer, SensorManager.SENSOR_DELAY_NORMAL);


                start_stop = (Button) findViewById(R.id.start_stop);
                start_stop.setOnClickListener(new Button.OnClickListener() {
                    @Override
                    public void onClick(View arg0) {
                        if (start_stop.getText().equals("Start!")) {
                            senSensorManager.registerListener(MainActivity.this, senAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
                            senSensorManager.registerListener(MainActivity.this, senMagnetometer, SensorManager.SENSOR_DELAY_NORMAL);
                            senSensorManager.registerListener(MainActivity.this, senGyrometer, SensorManager.SENSOR_DELAY_NORMAL);
                            getSupportActionBar().hide();
                            appStarted = true;
                            start_stop.setText("Stop!");
                            start_stop.setBackgroundColor(Color.parseColor("#ffcc0000"));
                        } else {
                            getSupportActionBar().show();
                            appStarted = false;
                            senSensorManager.unregisterListener(MainActivity.this);
                            start_stop.setText("Start!");
                            start_stop.setBackgroundColor(Color.parseColor("#ff669900"));
                        }
                    }
                });
            }
        }
    }

    // Send message if permission to SMS is granted on the device.
    public void notifyEmergencyContacts() {
        Toast toast = Toast.makeText(this, "ALERT", Toast.LENGTH_LONG);
        toast.setGravity(Gravity.CENTER, 0, 0);
        toast.show();
        MediaPlayer mp = MediaPlayer.create(getApplicationContext(), R.raw.buzzsound);
        if (mp != null) {
            if (mp.isPlaying())
                mp.stop();
        }
        mp.start();
        SmsManager sms = SmsManager.getDefault();
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            missingPermissions();
        }
        mFusedLocationClient.getLastLocation()
                .addOnSuccessListener(this, new OnSuccessListener<Location>() {
                    @Override
                    public void onSuccess(Location location) {
                        // Got last known location. In some rare situations this can be null.
                        if (location != null) {
                            locationGPS = location;
                        }
                    }
                });
        for(int i=0;i<3;i++) {

            if( locationGPS!=null) {

                StringBuffer smsBody = new StringBuffer();
                smsBody.append(message + "\n" + "http://maps.google.com?q=");
                smsBody.append(locationGPS.getLatitude());
                smsBody.append(",");
                smsBody.append(locationGPS.getLongitude());
                sms.sendTextMessage(phoneNo.get(i), null, smsBody.toString(), null, null);
            }else{
                sms.sendTextMessage(phoneNo.get(i), null, message, null, null);
            }
        }
        while(mp.isPlaying()){
            //do nothing
        }
        mp.stop();
        mp.reset();
        mp.release();
        mp=null;
    }


    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        count++; //To be removed later
        Sensor mySensor = sensorEvent.sensor;
        if(appStarted) {
            SensorEventLoggerTask async = new SensorEventLoggerTask(this);
            async.execute(sensorEvent);
        }

    }

    private class SensorEventLoggerTask extends AsyncTask<SensorEvent, Void, Boolean> {

        private MainActivity myContextRef;

        public SensorEventLoggerTask(MainActivity myContextRef) {
            this.myContextRef = myContextRef;
        }

        @Override
        protected Boolean doInBackground(SensorEvent... events) {
            String FILENAME = "sensor_cache.txt";

            File file = new File(myContextRef.getApplicationContext().getCacheDir(), FILENAME);
            SensorEvent event = events[0];
            if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
                acc_x = event.values[0];
                acc_y = event.values[1];
                acc_z = event.values[2];

            }
            if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
                gyro_x = event.values[0];
                gyro_y = event.values[1];
                gyro_z = event.values[2];

            }
            if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
                mag_x = event.values[0];
                mag_y = event.values[1];
                mag_z = event.values[2];

            }

            if (!file.exists()) {
                try{
                    file.createNewFile();
                }catch (Exception e){
                    Log.d("ERROR CREATE FILE-MAIN",e.getMessage());
                }
            }
            try {
                Writer mwriter = new BufferedWriter(new FileWriter(file, true));
                String temp_sensor_data = System.currentTimeMillis() + "," + acc_x + "," + acc_y + "," + acc_z + "," + gyro_x + "," + gyro_y + "," + gyro_z + "," + mag_x + "," + mag_y + "," + mag_z+"\n";

                mwriter.write(temp_sensor_data);
                mwriter.close();
            } catch (IOException e) {
                Log.d("Controller", e.getMessage() + e.getLocalizedMessage() + e.getCause());
            }

            //Logic to implement distraction detection

            if (count % 1000 == 0) {

                file = new File(myContextRef.getApplicationContext().getCacheDir(), FILENAME);

                if(file.exists()) {
                    StringBuilder text = new StringBuilder();

                    try {
                        BufferedReader br = new BufferedReader(new FileReader(file));
                        String line;

                        while ((line = br.readLine()) != null) {
                            text.append(line);
                            text.append('\n');
                        }
                        br.close();
                    }
                    catch (IOException e) {
                        Log.d("Reading file", "ERROR" +e.getMessage());

                    }
                    Log.d("BEFORE DELETION Count", String.valueOf(count));
                    Log.d("BEFORE DELETION", String.valueOf(text));
                    boolean deleted = file.delete();
                }
                return true;
            }
            return false;
        }

        @Override
        protected void onPostExecute(Boolean result) {
            if (result != null && result== true) {
                myContextRef.notifyEmergencyContacts();
            }
        }
    }
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    protected void onPause() {
        super.onPause();
        senSensorManager.unregisterListener(MainActivity.this);
    }

    protected void onResume() {
        super.onResume();
        if (MainActivity.this != null){
        senSensorManager.registerListener(MainActivity.this, senAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);}
        senSensorManager.registerListener(MainActivity.this, senMagnetometer ,SensorManager.SENSOR_DELAY_NORMAL);
        senSensorManager.registerListener(MainActivity.this, senGyrometer ,SensorManager.SENSOR_DELAY_NORMAL);
    }

    // Menu related methods
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_main, menu); //your file name
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(final MenuItem item) {

        Intent i = new Intent(this, Contacts.class);
        startActivity(i);
        return true;
    }

    // Permissions related methods

    public boolean CheckPermission(Context context, String Permission) {

        if (ContextCompat.checkSelfPermission(context,
                Permission) == PackageManager.PERMISSION_GRANTED) {
                return true;
        } else {
            return false;
        }
    }


}
