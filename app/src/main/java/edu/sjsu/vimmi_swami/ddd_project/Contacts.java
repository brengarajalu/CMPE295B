package edu.sjsu.vimmi_swami.ddd_project;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.database.Cursor;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.net.Uri;
import android.os.Bundle;
import android.provider.ContactsContract;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;
import java.util.List;

/**
 * Created by VimmiRao on 10/16/2017.
 */

public class Contacts extends Activity {

    private static final int  MY_PERMISSIONS_REQUEST_READ_CONTACTS =0;
    private String contactNumber1=null,contactNumber2=null,contactNumber3=null;
    private Button buttonPickContact1,buttonPickContact2,buttonPickContact3,submit,cancel;
    private static final String FILENAME = "contacts.txt";
    private Boolean exists =false;
    List<String> phoneNo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
         addContacts();
    }

    private String getName(String contact){
        String[] cntct = contact.split(":");
        return cntct[0];
    }
    protected void addContacts(){


        setContentView(R.layout.contact);

        buttonPickContact1 = (Button)findViewById(R.id.pickcontact1);
        buttonPickContact2 = (Button)findViewById(R.id.pickcontact2);
        buttonPickContact3 = (Button)findViewById(R.id.pickcontact3);
        submit = (Button)findViewById(R.id.submit);
        cancel = (Button)findViewById(R.id.cancel);

        //If contacts file exists ,then the intent is started from Menu button
        File file = new File(getApplicationContext().getApplicationContext().getCacheDir(), FILENAME);

        if(file.exists()) {
            exists =true;
            try {
                FileInputStream fin = new FileInputStream(file);
                String ret = MainActivity.convertStreamToString(fin);

                //Make sure you close all streams.
                fin.close();
                phoneNo = Arrays.asList(ret.split(","));
                contactNumber1 = phoneNo.get(0);
                buttonPickContact1.setText(getName(contactNumber1));
                buttonPickContact1.setBackgroundColor(Color.parseColor("#ff0099cc"));
                contactNumber2 = phoneNo.get(1);
                buttonPickContact2.setText(getName(contactNumber2));
                buttonPickContact2.setBackgroundColor(Color.parseColor("#ff0099cc"));
                contactNumber3 = phoneNo.get(2);
                buttonPickContact3.setText(getName(contactNumber3));
                buttonPickContact3.setBackgroundColor(Color.parseColor("#ff0099cc"));
            } catch (Exception e) {
                Log.e("Read Contacts", e.getMessage());
            }
        }
        buttonPickContact1.setOnClickListener(new Button.OnClickListener()
        {
            @Override
            public void onClick(View arg0) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType(ContactsContract.CommonDataKinds.Phone.CONTENT_ITEM_TYPE);
                startActivityForResult(intent, 1);
            }
        });

        buttonPickContact2.setOnClickListener(new Button.OnClickListener()
        {
            @Override
            public void onClick(View arg0) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType(ContactsContract.CommonDataKinds.Phone.CONTENT_ITEM_TYPE);
                startActivityForResult(intent, 2);
            }
        });

        buttonPickContact3.setOnClickListener(new Button.OnClickListener()
        {
            @Override
            public void onClick(View arg0) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType(ContactsContract.CommonDataKinds.Phone.CONTENT_ITEM_TYPE);
                startActivityForResult(intent, 3);
            }
        });

        cancel.setOnClickListener(new Button.OnClickListener()
        {
            @Override
            public void onClick(View arg0) {
                if (exists){
                    Intent i=new Intent(Contacts.this, MainActivity.class);
                    i.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                    startActivity(i);
                }
                else{
                    Intent i=new Intent(Contacts.this, MainActivity.class);
                    i.putExtra("EXIT", true);
                    i.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                    finish();
                    startActivity(i);

                }

            }
        });

        submit.setOnClickListener(new Button.OnClickListener()
        {
            @Override
            public void onClick(View arg0) {
                if (contactNumber1==null||contactNumber2==null||contactNumber3==null){
                    Toast toast= Toast.makeText(Contacts.this, "Please choose 3 contacts!", Toast.LENGTH_SHORT);
                    toast.setGravity(Gravity.CENTER, 0, 0);
                    toast.show();
                }
                else if(contactNumber1.equals(contactNumber2) || contactNumber1.equals(contactNumber3)|| contactNumber2.equals(contactNumber3)){
                    Toast toast=Toast.makeText(Contacts.this, "Please choose different contacts!", Toast.LENGTH_SHORT);
                    toast.setGravity(Gravity.CENTER, 0, 0);
                    toast.show();

                }
                else{


                    File file = new File(getApplicationContext().getCacheDir(), FILENAME);

                    if(file.exists()) {
                        file.delete();

                    }
                    try{
                        file.createNewFile();
                    }catch(Exception e){
                        Log.d("ERROR CREATNG FILE-Con",e.getMessage());
                    }

                    try {
                        Writer mwriter = new BufferedWriter(new FileWriter(file));
                        String contacts= contactNumber1+","+contactNumber2+","+contactNumber3;
                        Log.d("CONTACTS SAVED",contacts );
                        mwriter.write(contacts);
                        mwriter.close();
                    } catch (IOException e) {
                        Log.d("Contacts Storage", e.getMessage() + e.getLocalizedMessage() + e.getCause());
                    }
                    Intent i=new Intent(Contacts.this, MainActivity.class);
                    i.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                    startActivity(i);
                }
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        // TODO Auto-generated method stub
        super.onActivityResult(requestCode, resultCode, data);

        if(resultCode == RESULT_OK){
                Uri contactData = data.getData();
                Cursor cursor =   getContentResolver().query(contactData, null, null, null, null);
                cursor.moveToFirst();

                String num = cursor.getString(cursor.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER));
                String name = cursor.getString(cursor.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME));
                String number = name+":"+num;
                if(requestCode == 1) {
                    Log.d("contact1:", number);
                    contactNumber1 = number;
                    buttonPickContact1.setText(name);
                    buttonPickContact1.setBackgroundColor(Color.parseColor("#ff99cc00"));
                } else if (requestCode == 2) {
                    Log.d("contact2:", number);
                    contactNumber2 = number;
                    buttonPickContact2.setText(name);
                    buttonPickContact2.setBackgroundColor(Color.parseColor("#ff99cc00"));
                } else if (requestCode == 3) {
                    Log.d("contact3:", number);
                    contactNumber3 = number;
                    buttonPickContact3.setText(name);
                    buttonPickContact3.setBackgroundColor(Color.parseColor("#ff99cc00"));
                }
        }
    }
}

