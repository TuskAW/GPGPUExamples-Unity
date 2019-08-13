using UnityEngine;

//========================================
// Solve the poisson equation L*v=b
// using Conjugate Gradient method
//========================================
//----- Conjugate Gradient algorithm -----
//
// Solve the linear equation: L*v=b
// Coefficient matrix (Laplacian): L
// Solution vector: v
// Source term vector: b
//
// Initialization
// set initial guess v0
// r0 = b - L*v0
// p0 = r0
//
// Iteration ( until |rk|/|b|<eps )
// alpha = (rk,rk)/(pk,L*pk)
// vk1 = vk + alpha*pk
// rk1 = rk - alpha*L*pk
// beta = (rk1,rk1)/(rk,rk)
// pk1 = rk1 + beta*pk
//
//----------------------------------------
namespace GPGPUExamples
{
    public class Poisson2D_GPU : MonoBehaviour
    {
        [SerializeField] ComputeShader _computeShader;
        [SerializeField] ComputeShader _computeShaderForTest;
        [SerializeField] Material _debugTextureMaterial;
        [SerializeField] int _width = 100;
        [SerializeField] int _height = 100;
        [SerializeField] float _delta = 0.1f;
        [SerializeField] int _maxIter = 500;
        [SerializeField] float _eps = 1.0e-7f;

        Vector2Int _gpuThreads = new Vector2Int(16, 16);

        bool _initialized = false;

        int _numOfGroupsX, _numOfGroupsY;

        float[] _partialDot;
        ComputeBuffer _partialDotBuffer;

        RenderTexture _b; // source term
        RenderTexture _vk; // the solution of the poisson equation
        RenderTexture _vk1;
        RenderTexture _rka;
        RenderTexture _rkb;
        RenderTexture _pka;
        RenderTexture _pkb;
        RenderTexture _Lp;
        RenderTexture _Lv;

        float[] _rtOutBufferArray;
        ComputeBuffer _rtOutBuffer;

        void Start()
        {
            if (!SystemInfo.supportsComputeShaders)
            {
                Debug.LogError("Compute Shader is not Support!!");
                return;
            }
            if (_computeShader == null)
            {
                Debug.LogError("Compute Shader has not been assigned!!");
                return;
            }

            InitializeForGPUMode();

            System.Diagnostics.Stopwatch _stopWatch = new System.Diagnostics.Stopwatch();
            _stopWatch.Start();

            SetTestData();
            SolvePoissonGPU(_width, _height, _delta);

            _stopWatch.Stop();
            Debug.Log("Elapsed time: " + _stopWatch.ElapsedMilliseconds + "ms");

            OutputRTValue(_b);
            DataExporter.Export("PoissonSource", _rtOutBufferArray, _width, _height);
            OutputRTValue(_vk);
            DataExporter.Export("PoissonResult", _rtOutBufferArray, _width, _height);
        }

        void Update()
        {
            if (!SystemInfo.supportsComputeShaders)
            {
                Debug.LogError("Compute Shader is not Support!!");
                return;
            }
            if (_computeShader == null)
            {
                Debug.LogError("Compute Shader has not been assigned!!");
                return;
            }

            // System.Diagnostics.Stopwatch _stopWatch = new System.Diagnostics.Stopwatch();
            // _stopWatch.Start();

            // SetTestData();
            // SolvePoissonGPU(_width, _height, _delta);

            // _stopWatch.Stop();
            // Debug.Log("Elapsed time: " + _stopWatch.ElapsedMilliseconds + "ms");
        }

        void SetTestData()
        {
            if (!_initialized)
            {
                InitializeForGPUMode();
            }

            int kernelID = _computeShaderForTest.FindKernel("CsSetSourceTermAndInitialGuess");
            _computeShaderForTest.SetInt("_width", _width);
            _computeShaderForTest.SetInt("_height", _height);
            _computeShaderForTest.SetTexture(kernelID, "_v", _vk); // Initial guess
            _computeShaderForTest.SetTexture(kernelID, "_sourceTerm", _b); // Source term
            _computeShaderForTest.Dispatch(kernelID, Mathf.CeilToInt((float)_width / _gpuThreads.x), 
                                                Mathf.CeilToInt((float)_height / _gpuThreads.y), 1);
        }

        void OutputRTValue(RenderTexture rt)
        {
            int kernelID = _computeShaderForTest.FindKernel("CsOutputRTValue");
            _computeShaderForTest.SetInt("_width", _width);
            _computeShaderForTest.SetInt("_height", _height);
            _computeShaderForTest.SetTexture(kernelID, "_TempInputTex", rt);
            _computeShaderForTest.SetBuffer(kernelID, "_OutComputeBuffer", _rtOutBuffer);
            _computeShaderForTest.Dispatch(kernelID, Mathf.CeilToInt((float)_width / _gpuThreads.x), 
                                                    Mathf.CeilToInt((float)_height / _gpuThreads.y), 1);
            _rtOutBuffer.GetData(_rtOutBufferArray);
        }

        void InitializeForGPUMode()
        {
            _numOfGroupsX = Mathf.CeilToInt((float)_width / _gpuThreads.x);
            _numOfGroupsY = Mathf.CeilToInt((float)_height / _gpuThreads.y);

            _partialDot = new float[_numOfGroupsX*_numOfGroupsY];
            _partialDotBuffer = new ComputeBuffer(_numOfGroupsX*_numOfGroupsY, sizeof(float));

            _b = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _b.enableRandomWrite = true;
            _b.Create();

            _vk = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _vk.enableRandomWrite = true;
            _vk.Create();

            _vk1 = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _vk1.enableRandomWrite = true;
            _vk1.Create();

            _rka = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _rka.enableRandomWrite = true;
            _rka.Create();

            _rkb = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _rkb.enableRandomWrite = true;
            _rkb.Create();

            _pka = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _pka.enableRandomWrite = true;
            _pka.Create();

            _pkb = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _pkb.enableRandomWrite = true;
            _pkb.Create();

            _Lp = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _Lp.enableRandomWrite = true;
            _Lp.Create();

            _Lv = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _Lv.enableRandomWrite = true;
            _Lv.Create();

            _rtOutBufferArray = new float[_width*_height];
            _rtOutBuffer = new ComputeBuffer(_width*_height, sizeof(float));

            _initialized = true;
        }

        void SolvePoissonGPU(int width, int height, float delta)
        {
            if (!_initialized)
            {
                InitializeForGPUMode();
            }

            // r0 = b - L*v0
            CsLpMV(_Lv, _vk, width, height, delta);
            CsSzaxpy(_rka, -1.0f, _Lv, _b, width, height);
            CsCopyVector(_rkb, _rka, width, height);

            // p0 = r0
            CsCopyVector(_pka, _rka, width, height);

            // (r0, r0)
            float rk1rk1 = CsDotProduct(_rka, _rkb, width, height);

            // Iteration
            for (int iter = 0; iter < _maxIter; iter++)
            {
                float rkrk = rk1rk1;

                // Convergence check
                if (rkrk < _eps)
                {
                    Debug.Log("Converged");
                    Debug.Log("Iter: " + iter);
                    Debug.Log("rkrk: " + rkrk);
                    break;
                }

                // (pk, L*pk)
                CsLpMV(_Lp, _pka, width, height, delta);
                float pLp = CsDotProduct(_pka, _Lp, width, height);

                // alpha = (rk,rk)/(pk,L*pk)
                float alpha = rkrk/pLp;

                // vk1 = vk + alpha*pk
                CsSzaxpy(_vk1,  alpha, _pka, _vk, width, height);
                CsCopyVector(_vk, _vk1, width, height);
                
                // rk1 = rk - alpha*L*pk
                CsSzaxpy(_rkb, -alpha, _Lp, _rka, width, height);
                CsCopyVector(_rka, _rkb, width, height);

                // beta = (rk1,rk1)/(rk,rk)
                rk1rk1 = CsDotProduct(_rka, _rkb, width, height);
                float beta = rk1rk1/rkrk;

                // pk1 = rk1 + beta*pk
                CsSzaxpy(_pkb, beta, _pka, _rka, width, height);
                CsCopyVector(_pka, _pkb, width, height);
            }
        }

        void CsCopyVector(RenderTexture dst, RenderTexture src, int width, int height)
        {
            int kernelID = _computeShader.FindKernel("CsCopyVector");
            _computeShader.SetInt("_width", width);
            _computeShader.SetInt("_height", height);
            _computeShader.SetTexture(kernelID, "_x", src);
            _computeShader.SetTexture(kernelID, "_y", dst);
            _computeShader.Dispatch(kernelID, Mathf.CeilToInt((float)width / _gpuThreads.x), 
                                                Mathf.CeilToInt((float)height / _gpuThreads.y), 1);
        }

        void CsLpMV(RenderTexture Lv, RenderTexture vk, int width, int height, float h)
        {
            int kernelID = _computeShader.FindKernel("CsLpMV");
            _computeShader.SetInt("_width", width);
            _computeShader.SetInt("_height", height);
            _computeShader.SetFloat("_h", h);
            _computeShader.SetTexture(kernelID, "_v", vk);
            _computeShader.SetTexture(kernelID, "_Lv", Lv);
            _computeShader.Dispatch(kernelID, Mathf.CeilToInt((float)width / _gpuThreads.x), 
                                                Mathf.CeilToInt((float)height / _gpuThreads.y), 1);
        }

        void CsSzaxpy(RenderTexture z, float alpha, RenderTexture x, RenderTexture y, int width, int height)
        {
            int kernelID = _computeShader.FindKernel("CsSzaxpy");
            _computeShader.SetInt("_width", width);
            _computeShader.SetInt("_height", height);
            _computeShader.SetFloat("_alpha", alpha);
            _computeShader.SetTexture(kernelID, "_x", x);
            _computeShader.SetTexture(kernelID, "_y", y);
            _computeShader.SetTexture(kernelID, "_z", z);
            _computeShader.Dispatch(kernelID, Mathf.CeilToInt((float)width / _gpuThreads.x), 
                                                Mathf.CeilToInt((float)height / _gpuThreads.y), 1);
        }

        float CsDotProduct(RenderTexture vectorA, RenderTexture vectorB, int width, int height)
        {
            int kernelID = _computeShader.FindKernel("CsDotProduct");
            _computeShader.SetInt("_width", width);
            _computeShader.SetInt("_height", height);
            _computeShader.SetInt("_numOfGroupsX", _numOfGroupsX);
            _computeShader.SetTexture(kernelID, "_a", vectorA);
            _computeShader.SetTexture(kernelID, "_b", vectorB);
            _computeShader.SetBuffer(kernelID, "_PartialDot", _partialDotBuffer);

            // The total number of execution threads is numOfGroups*numOfGpuThreads
            _computeShader.Dispatch(kernelID, Mathf.CeilToInt((float)width / _gpuThreads.x), 
                                                Mathf.CeilToInt((float)height / _gpuThreads.y), 1);
            _partialDotBuffer.GetData(_partialDot);

            float dotProduct = 0.0f;
            for (int k = 0; k < _numOfGroupsX*_numOfGroupsY; k++)
            {
                dotProduct += _partialDot[k];
            }
    
            return dotProduct;
        }
    }
}
