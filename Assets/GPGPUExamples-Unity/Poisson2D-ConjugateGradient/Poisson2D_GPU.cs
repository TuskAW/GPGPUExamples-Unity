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
        [SerializeField] int _width = 100;
        [SerializeField] int _height = 100;
        [SerializeField] float _dh = 0.01f;
        [SerializeField] int _maxIter = 50;

        Vector2Int _gpuThreads = new Vector2Int(16, 16);

        bool _initialized = false;

        int _numOfGroupsX, _numOfGroupsY;

        float[] _partialDot;
        ComputeBuffer _partialDotBuffer;

        RenderTexture _b; // source term
        RenderTexture _vk; // the solution of the poisson equation
        RenderTexture _rk;
        RenderTexture _pk;
        RenderTexture _Lp;
        RenderTexture _Lv;

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

            System.Diagnostics.Stopwatch _stopWatch = new System.Diagnostics.Stopwatch();
            _stopWatch.Start();

            SolvePoissonGPU(_width, _height, _dh);
            Debug.Log("WxH: " + _width*_height);

            _stopWatch.Stop();
            Debug.Log("Elapsed time: " + _stopWatch.ElapsedMilliseconds + "ms");
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

            _rk = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _rk.enableRandomWrite = true;
            _rk.Create();

            _pk = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _pk.enableRandomWrite = true;
            _pk.Create();

            _Lp = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _Lp.enableRandomWrite = true;
            _Lp.Create();

            _Lv = new RenderTexture(_width, _height, 0, RenderTextureFormat.RFloat);
            _Lv.enableRandomWrite = true;
            _Lv.Create();

            _initialized = true;
        }

        void SolvePoissonGPU(int width, int height, float h)
        {
            if (!_initialized)
            {
                InitializeForGPUMode();
            }

            // //***********************************************
            // int kernelID = _computeShader.FindKernel("CsSetInitialData");
            // _computeShader.SetInt("_width", width);
            // _computeShader.SetInt("_height", height);
            // _computeShader.SetTexture(kernelID, "_VectorA", _Lp);
            // _computeShader.SetTexture(kernelID, "_VectorB", _Lv);
            // _computeShader.Dispatch(kernelID, Mathf.CeilToInt((float)width / _gpuThreads.x), 
            //                                     Mathf.CeilToInt((float)height / _gpuThreads.y), 1);

            // float dot = CsDotProduct(_Lp, _Lv, width, height);
            // Debug.Log("Dot: " + dot);
            // //***********************************************

            CsSetInitialData(_vk, _b, width, height);
            CsLpMV(_Lv, _vk, width, height, h);
            CsSzaxpy(_rk, -1.0f, _Lv, _b, width, height);
            CsCopyVector(_pk, _rk, width, height);

            // Iteration
            for (int iter = 0; iter < _maxIter; iter++)
            {
                float rkrk = CsDotProduct(_rk, _rk, width, height);

                // Convergence check
                if (rkrk < Mathf.Epsilon)
                {
                    Debug.Log("Converged");
                    Debug.Log("Iter: " + iter);
                    Debug.Log("rkrk: " + rkrk);
                    break;
                }

                CsLpMV(_Lp, _pk, width, height, h);
                float pLp = CsDotProduct(_pk, _Lp, width, height);
                float alpha = rkrk/pLp;

                CsSzaxpy(_rk, -alpha, _Lp, _rk, width, height);

                float beta = CsDotProduct(_rk, _rk, width, height)/rkrk;
                CsSzaxpy(_pk, beta, _pk, _rk, width, height);
            }
        }

        void CsSetInitialData(RenderTexture v0, RenderTexture b, int width, int height)
        {
            int kernelID = _computeShader.FindKernel("CsSetInitialData");
            _computeShader.SetInt("_width", width);
            _computeShader.SetInt("_height", height);
            _computeShader.SetTexture(kernelID, "_vk", v0); // Initial guess
            _computeShader.SetTexture(kernelID, "_b", b); // Source term
            _computeShader.Dispatch(kernelID, Mathf.CeilToInt((float)width / _gpuThreads.x), 
                                                Mathf.CeilToInt((float)height / _gpuThreads.y), 1);
        }

        void CsCopyVector(RenderTexture dst, RenderTexture src, int width, int height)
        {
            int kernelID = _computeShader.FindKernel("CsCopyVector");
            _computeShader.SetInt("_width", width);
            _computeShader.SetInt("_height", height);
            _computeShader.SetTexture(kernelID, "_y", src);
            _computeShader.SetTexture(kernelID, "_x", dst);
            _computeShader.Dispatch(kernelID, Mathf.CeilToInt((float)width / _gpuThreads.x), 
                                                Mathf.CeilToInt((float)height / _gpuThreads.y), 1);
        }

        void CsLpMV(RenderTexture Lv, RenderTexture vk, int width, int height, float h)
        {
            int kernelID = _computeShader.FindKernel("CsLpMV1");
            _computeShader.SetInt("_width", width);
            _computeShader.SetInt("_height", height);
            _computeShader.SetFloat("_h", h);
            _computeShader.SetTexture(kernelID, "_vk", vk);
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
            _computeShader.SetTexture(kernelID, "_VectorA", vectorA);
            _computeShader.SetTexture(kernelID, "_VectorB", vectorB);
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
