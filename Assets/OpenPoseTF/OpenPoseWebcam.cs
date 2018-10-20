#if !UNITY_WSA_10_0
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

#if UNITY_5_3 || UNITY_5_3_OR_NEWER
using UnityEngine.SceneManagement;
#endif
using OpenCVForUnity;


public class OpenPoseWebcam : MonoBehaviour
{
    const float inWidth = 368;
    const float inHeight = 368;

    //COCO
    Dictionary<string, int> BODY_PARTS
        = new Dictionary<string, int>() {
            { "Nose", 0 }, { "Neck", 1 }, { "RShoulder", 2 }, { "RElbow", 3 }, {
                "RWrist",
                4
            },
            { "LShoulder",5 }, { "LElbow", 6 }, { "LWrist", 7 }, { "RHip", 8 }, {
                "RKnee",
                9
            },
            { "RAnkle", 10 }, { "LHip", 11 }, { "LKnee", 12 }, { "LAnkle", 13 }, {
                "REye",
                14
            },
            { "LEye", 15 }, { "REar", 16 }, { "LEar", 17 }, {
                "Background",
                18
            }
        };

    string[,] POSE_PAIRS
    = new string[,] {
            { "Neck", "RShoulder" }, { "Neck", "LShoulder" }, {
                "RShoulder",
                "RElbow"
            },
            { "RElbow", "RWrist" }, { "LShoulder", "LElbow" }, {
                "LElbow",
                "LWrist"
            },
            { "Neck", "RHip" }, { "RHip", "RKnee" }, { "RKnee", "RAnkle" }, {
                "Neck",
                "LHip"
            },
            { "LHip", "LKnee" }, { "LKnee", "LAnkle" }, { "Neck", "Nose" }, {
                "Nose",
                "REye"
            },
            { "REye", "REar" }, { "Nose", "LEye" }, { "LEye", "LEar" }
    };
    
    string graph_filepath;
    Net net = null;
    WebCamTextureToMatHelper webCamTextureToMatHelper;

    void Start()
    {
        graph_filepath = Utils.getFilePath("dnn/graph1.pb");
        webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();
        webCamTextureToMatHelper.Initialize();
        Mat img = webCamTextureToMatHelper.GetMat();

        if (!string.IsNullOrEmpty(graph_filepath))
        {
            net = Dnn.readNetFromTensorflow(graph_filepath);
        }

        if (net == null)
        {
            Imgproc.putText(img, "model file is not loaded.", new Point(5, img.rows() - 30), Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2, Imgproc.LINE_AA, false);
        }

        float imageWidth = img.width();
        float imageHeight = img.height();
        float widthScale = (float)Screen.width / imageWidth;
        float heightScale = (float)Screen.height / imageHeight;
        if (widthScale < heightScale)
        {
            Camera.main.orthographicSize = (imageWidth * (float)Screen.height / (float)Screen.width) / 2;
        }
        else
        {
            Camera.main.orthographicSize = imageHeight / 2;
        }
    }

    void Run()
    {
        Utils.setDebugMode(true);

        Mat img = webCamTextureToMatHelper.GetMat();
        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGBA2BGR);
        gameObject.transform.localScale = new Vector3(img.width(), img.height(), 1);

        if(net!=null)
        {
            float frameWidth = img.cols();
            float frameHeight = img.rows();
            Mat input = Dnn.blobFromImage(img, 1.0, new Size(inWidth, inHeight), new Scalar(0, 0, 0), false, false);
            net.setInput(input,"image");
            Mat output = net.forward("Openpose/concat_stage7");
            
            output = output.reshape(1, 57);
            List<Point> points = new List<Point>();
            for (int i = 0; i < BODY_PARTS.Count; i++)
            {
                Mat heatMap = output.row(i).reshape(1, 46);
                Core.MinMaxLocResult result = Core.minMaxLoc(heatMap);
                heatMap.Dispose();
                
                double x = (frameWidth * result.maxLoc.x) / 46;
                double y = (frameHeight * result.maxLoc.y) / 46;
                
                if (result.maxVal > 0.3)
                {
                    points.Add(new Point(x, y));

                }
                else
                {
                    points.Add(null);
                }

            }

            for (int i = 0; i < POSE_PAIRS.GetLength(0); i++)
            {
                string partFrom = POSE_PAIRS[i, 0];
                string partTo = POSE_PAIRS[i, 1];

                int idFrom = BODY_PARTS[partFrom];
                int idTo = BODY_PARTS[partTo];

                if (points[idFrom] != null && points[idTo] != null)
                {
                    Debug.Log("x=" + points[idFrom].x + " y=" + points[idFrom].y);
                    Imgproc.line(img, points[idFrom], points[idTo], new Scalar(0, 255, 0), 3);
                    Imgproc.ellipse(img, points[idFrom], new Size(3, 3), 0, 0, 360, new Scalar(0, 0, 255), Core.FILLED);
                    Imgproc.ellipse(img, points[idTo], new Size(3, 3), 0, 0, 360, new Scalar(0, 0, 255), Core.FILLED);
                    float avgFrameRate = Time.frameCount / Time.time;
                    Imgproc.putText(img, "FR="+avgFrameRate, new Point(5, img.rows() - 30), Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                }
            }
         }

        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);
        Texture2D texture = new Texture2D(img.cols(), img.rows(), TextureFormat.RGBA32, false);
        Utils.matToTexture2D(img, texture);
        gameObject.GetComponent<Renderer>().material.mainTexture = texture;

    }

    void Update()
    {
        if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
        {
            Run();
        }
    }

    public void Back()
    {
        SceneManager.LoadScene("Menu");
    }
}
#endif