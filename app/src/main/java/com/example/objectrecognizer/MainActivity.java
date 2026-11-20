package com.example.objectrecognizer;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String MODEL_FILE = "efficientnet_lite0.tflite";

    private PreviewView previewView;
    private TextView resultText;
    private ImageClassifier classifier;
    private ExecutorService cameraExecutor;

    private final ActivityResultLauncher<String> permissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) startCamera();
                else resultText.setText(R.string.no_permission);
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        resultText = findViewById(R.id.resultText);
        cameraExecutor = Executors.newSingleThreadExecutor();

        try {
            ImageClassifier.ImageClassifierOptions options =
                    ImageClassifier.ImageClassifierOptions.builder()
                            .setMaxResults(3)
                            .setScoreThreshold(0.3f)
                            .build();
            classifier = ImageClassifier.createFromFileAndOptions(this, MODEL_FILE, options);
        } catch (Exception e) {
            resultText.setText("Ошибка загрузки модели (" + MODEL_FILE + "): " + e.getMessage());
            return;
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                ProcessCameraProvider provider = future.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build();

                analysis.setAnalyzer(cameraExecutor, image -> {
                    Bitmap bitmap = image.toBitmap();
                    int rotation = image.getImageInfo().getRotationDegrees();
                    image.close();

                    ImageProcessingOptions opts = ImageProcessingOptions.builder()
                            .setOrientation(toOrientation(rotation))
                            .build();

                    List<Classifications> results =
                            classifier.classify(TensorImage.fromBitmap(bitmap), opts);
                    showResults(results);
                });

                provider.unbindAll();
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                        preview, analysis);
            } catch (Exception e) {
                resultText.setText("Ошибка камеры: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private ImageProcessingOptions.Orientation toOrientation(int rotation) {
        switch (rotation) {
            case 90:  return ImageProcessingOptions.Orientation.RIGHT_TOP;
            case 180: return ImageProcessingOptions.Orientation.BOTTOM_RIGHT;
            case 270: return ImageProcessingOptions.Orientation.LEFT_BOTTOM;
            default:  return ImageProcessingOptions.Orientation.TOP_LEFT;
        }
    }

    private void showResults(List<Classifications> results) {
        StringBuilder sb = new StringBuilder();
        if (!results.isEmpty()) {
            for (Category c : results.get(0).getCategories()) {
                sb.append(String.format(Locale.US, "%s — %.0f%%\n",
                        c.getLabel(), c.getScore() * 100));
            }
        }
        String text = sb.length() == 0 ? getString(R.string.not_recognized) : sb.toString().trim();
        runOnUiThread(() -> resultText.setText(text));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        if (classifier != null) classifier.close();
    }
}
