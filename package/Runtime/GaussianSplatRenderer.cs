// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using Unity.Jobs;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Assertions;

namespace GaussianSplatting.Runtime
{
    class GaussianSplatRenderSystem
    {
        // ReSharper disable MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal static readonly ProfilerMarker s_ProfDraw = new(ProfilerCategory.Render, "GaussianSplat.Draw", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCompose = new(ProfilerCategory.Render, "GaussianSplat.Compose", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCalcView = new(ProfilerCategory.Render, "GaussianSplat.CalcView", MarkerFlags.SampleGPU);
        // ReSharper restore MemberCanBePrivate.Global

        public static GaussianSplatRenderSystem instance => ms_Instance ??= new GaussianSplatRenderSystem();
        static GaussianSplatRenderSystem ms_Instance;

        readonly Dictionary<GaussianSplatRenderer, MaterialPropertyBlock> m_Splats = new();
        readonly HashSet<Camera> m_CameraCommandBuffersDone = new();
        readonly List<(GaussianSplatRenderer, MaterialPropertyBlock)> m_ActiveSplats = new();

        CommandBuffer m_CommandBuffer;

        public void RegisterSplat(GaussianSplatRenderer r)
        {
            if (m_Splats.Count == 0)
            {
                if (GraphicsSettings.currentRenderPipeline == null)
                    Camera.onPreCull += OnPreCullCamera;
            }

            m_Splats.Add(r, new MaterialPropertyBlock());
        }

        public void UnregisterSplat(GaussianSplatRenderer r)
        {
            if (!m_Splats.ContainsKey(r))
                return;
            m_Splats.Remove(r);
            if (m_Splats.Count == 0)
            {
                if (m_CameraCommandBuffersDone != null)
                {
                    if (m_CommandBuffer != null)
                    {
                        foreach (var cam in m_CameraCommandBuffersDone)
                        {
                            if (cam)
                                cam.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                        }
                    }
                    m_CameraCommandBuffersDone.Clear();
                }

                m_ActiveSplats.Clear();
                m_CommandBuffer?.Dispose();
                m_CommandBuffer = null;
                Camera.onPreCull -= OnPreCullCamera;
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public bool GatherSplatsForCamera(Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return false;
            // gather all active & valid splat objects
            m_ActiveSplats.Clear();
            foreach (var kvp in m_Splats)
            {
                var gs = kvp.Key;
                if (gs == null || !gs.isActiveAndEnabled || !gs.HasValidAsset || !gs.HasValidRenderSetup)
                    continue;
                m_ActiveSplats.Add((kvp.Key, kvp.Value));
            }
            if (m_ActiveSplats.Count == 0)
                return false;

            // sort them by depth from camera
            var camTr = cam.transform;
            m_ActiveSplats.Sort((a, b) =>
            {
                var trA = a.Item1.transform;
                var trB = b.Item1.transform;
                var posA = camTr.InverseTransformPoint(trA.position);
                var posB = camTr.InverseTransformPoint(trB.position);
                return posA.z.CompareTo(posB.z);
            });

            return true;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public Material SortAndRenderSplats(Camera cam, CommandBuffer cmb)
        {
            Material matComposite = null;
            foreach (var kvp in m_ActiveSplats)
            {
                var gs = kvp.Item1;
                matComposite = gs.m_MatComposite;
                var mpb = kvp.Item2;

                // sort
                var matrix = gs.transform.localToWorldMatrix;
                if (gs.m_FrameCounter % gs.m_SortNthFrame == 0)
                    gs.SortPoints(cmb, cam, matrix);
                ++gs.m_FrameCounter;

                // cache view
                kvp.Item2.Clear();
                Material displayMat = gs.m_RenderMode switch
                {
                    GaussianSplatRenderer.RenderMode.DebugPoints => gs.m_MatDebugPoints,
                    GaussianSplatRenderer.RenderMode.DebugPointIndices => gs.m_MatDebugPoints,
                    GaussianSplatRenderer.RenderMode.DebugBoxes => gs.m_MatDebugBoxes,
                    GaussianSplatRenderer.RenderMode.DebugChunkBounds => gs.m_MatDebugBoxes,
                    _ => gs.m_MatSplats
                };
                if (displayMat == null)
                    continue;

                gs.SetAssetDataOnMaterial(mpb);
                mpb.SetBuffer(GaussianSplatRenderer.Props.SplatChunks, gs.m_GpuChunks);

                mpb.SetBuffer(GaussianSplatRenderer.Props.SplatViewData, gs.m_GpuView);

                mpb.SetBuffer(GaussianSplatRenderer.Props.OrderBuffer, gs.m_GpuSortKeys);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatScale, gs.m_SplatScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatOpacityScale, gs.m_OpacityScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatSize, gs.m_PointDisplaySize);
                mpb.SetInteger(GaussianSplatRenderer.Props.SHOrder, gs.m_SHOrder);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayIndex, gs.m_RenderMode == GaussianSplatRenderer.RenderMode.DebugPointIndices ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayChunks, gs.m_RenderMode == GaussianSplatRenderer.RenderMode.DebugChunkBounds ? 1 : 0);

                cmb.BeginSample(s_ProfCalcView);
                gs.CalcViewData(cmb, cam, matrix);
                cmb.EndSample(s_ProfCalcView);

                // draw
                int indexCount = 6;
                int instanceCount = gs.splatCount;
                MeshTopology topology = MeshTopology.Triangles;
                if (gs.m_RenderMode is GaussianSplatRenderer.RenderMode.DebugBoxes or GaussianSplatRenderer.RenderMode.DebugChunkBounds)
                    indexCount = 36;
                if (gs.m_RenderMode == GaussianSplatRenderer.RenderMode.DebugChunkBounds)
                    instanceCount = gs.m_GpuChunksValid ? gs.m_GpuChunks.count : 0;

                cmb.BeginSample(s_ProfDraw);
                cmb.DrawProcedural(gs.m_GpuIndexBuffer, matrix, displayMat, 0, topology, indexCount, instanceCount, mpb);
                cmb.EndSample(s_ProfDraw);
            }
            return matComposite;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        // ReSharper disable once UnusedMethodReturnValue.Global - used by HDRP/URP features that are not always compiled
        public CommandBuffer InitialClearCmdBuffer(Camera cam)
        {
            m_CommandBuffer ??= new CommandBuffer {name = "RenderGaussianSplats"};
            if (GraphicsSettings.currentRenderPipeline == null && cam != null && !m_CameraCommandBuffersDone.Contains(cam))
            {
                cam.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                m_CameraCommandBuffersDone.Add(cam);
            }

            // get render target for all splats
            m_CommandBuffer.Clear();
            return m_CommandBuffer;
        }

        void OnPreCullCamera(Camera cam)
        {
            if (!GatherSplatsForCamera(cam))
                return;

            InitialClearCmdBuffer(cam);

            m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT, -1, -1, 0, FilterMode.Point, GraphicsFormat.R16G16B16A16_SFloat);
            m_CommandBuffer.SetRenderTarget(GaussianSplatRenderer.Props.GaussianSplatRT, BuiltinRenderTextureType.CurrentActive);
            m_CommandBuffer.ClearRenderTarget(RTClearFlags.Color, new Color(0, 0, 0, 0), 0, 0);

            // add sorting, view calc and drawing commands for each splat object
            Material matComposite = SortAndRenderSplats(cam, m_CommandBuffer);

            // compose
            m_CommandBuffer.BeginSample(s_ProfCompose);
            m_CommandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
            m_CommandBuffer.DrawProcedural(Matrix4x4.identity, matComposite, 0, MeshTopology.Triangles, 3, 1);
            m_CommandBuffer.EndSample(s_ProfCompose);
            m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT);
        }
    }

    [ExecuteInEditMode]
    public class GaussianSplatRenderer : MonoBehaviour
    {
        public enum RenderMode
        {
            Splats,
            DebugPoints,
            DebugPointIndices,
            DebugBoxes,
            DebugChunkBounds,
        }
        public GaussianSplatAsset m_Asset;

        [Range(0.1f, 2.0f)] [Tooltip("Additional scaling factor for the splats")]
        public float m_SplatScale = 1.0f;
        [Range(0.05f, 20.0f)]
        [Tooltip("Additional scaling factor for opacity")]
        public float m_OpacityScale = 1.0f;
        [Range(0, 3)] [Tooltip("Spherical Harmonics order to use")]
        public int m_SHOrder = 3;
        [Range(1,30)] [Tooltip("Sort splats only every N frames")]
        public int m_SortNthFrame = 1;

        public RenderMode m_RenderMode = RenderMode.Splats;
        [Range(1.0f,15.0f)] public float m_PointDisplaySize = 3.0f;

        public GaussianCutout[] m_Cutouts;

        public Shader m_ShaderSplats;
        public Shader m_ShaderComposite;
        public Shader m_ShaderDebugPoints;
        public Shader m_ShaderDebugBoxes;
        [Tooltip("Gaussian splatting compute shader")]
        public ComputeShader m_CSSplatUtilities;

        int m_SplatCount; // initially same as asset splat count, but editing can change this
        GraphicsBuffer m_GpuSortDistances;
        internal GraphicsBuffer m_GpuSortKeys;
        GraphicsBuffer m_GpuPosData;
        GraphicsBuffer m_GpuOtherData;
        GraphicsBuffer m_GpuSHData;
        GraphicsBuffer m_GpuColorData;

        // these buffers are only for splat editing->duplicate edit tool
        GraphicsBuffer m_GpuDupPosData;
        GraphicsBuffer m_GpuDupOtherData;
        GraphicsBuffer m_GpuDupSHData;
        GraphicsBuffer m_GpuDupColorData;
        int preDupSplatCount = 0;

        Texture m_GpuTexColorData;
        internal GraphicsBuffer m_GpuChunks;
        internal bool m_GpuChunksValid;
        internal GraphicsBuffer m_GpuView;
        internal GraphicsBuffer m_GpuIndexBuffer;

        // these buffers are only for splat editing, and are lazily created
        GraphicsBuffer m_GpuEditCutouts;
        GraphicsBuffer m_GpuEditCountsBounds;
        GraphicsBuffer m_GpuEditSelected;
        GraphicsBuffer m_GpuEditDeleted;
        GraphicsBuffer m_GpuEditSelectedMouseDown; // selection state at start of operation
        GraphicsBuffer m_GpuEditPosMouseDown; // position state at start of operation
        GraphicsBuffer m_GpuEditOtherMouseDown; // rotation/scale state at start of operation

        GpuSorting m_Sorter;
        GpuSorting.Args m_SorterArgs;

        internal Material m_MatSplats;
        internal Material m_MatComposite;
        internal Material m_MatDebugPoints;
        internal Material m_MatDebugBoxes;

        internal int m_FrameCounter;
        GaussianSplatAsset m_PrevAsset;
        Hash128 m_PrevHash;

        static readonly ProfilerMarker s_ProfSort = new(ProfilerCategory.Render, "GaussianSplat.Sort", MarkerFlags.SampleGPU);

        internal static class Props
        {
            public static readonly int SplatPos = Shader.PropertyToID("_SplatPos");
            public static readonly int SplatOther = Shader.PropertyToID("_SplatOther");
            public static readonly int SplatSH = Shader.PropertyToID("_SplatSH");
            public static readonly int SplatColor = Shader.PropertyToID("_SplatColor");
            public static readonly int SplatTexColor = Shader.PropertyToID("_SplatTexColor");
            public static readonly int SplatDuplicatedPos = Shader.PropertyToID("_SplatDuplicatedPos");
            public static readonly int SplatDuplicatedOther = Shader.PropertyToID("_SplatDuplicatedOther");
            public static readonly int SplatDuplicatedSH = Shader.PropertyToID("_SplatDuplicatedSH");
            public static readonly int SplatDuplicatedColor = Shader.PropertyToID("_SplatDuplicatedColor");
            public static readonly int DupSplatCount = Shader.PropertyToID("_DupSplatCount");
            public static readonly int SplatSelectedBits = Shader.PropertyToID("_SplatSelectedBits");
            public static readonly int SplatDeletedBits = Shader.PropertyToID("_SplatDeletedBits");
            public static readonly int SelectionMode = Shader.PropertyToID("_SelectionMode");
            public static readonly int SplatBitsValid = Shader.PropertyToID("_SplatBitsValid");
            public static readonly int SplatFormat = Shader.PropertyToID("_SplatFormat");
            public static readonly int SplatChunks = Shader.PropertyToID("_SplatChunks");
            public static readonly int SplatChunkCount = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int SplatViewData = Shader.PropertyToID("_SplatViewData");
            public static readonly int OrderBuffer = Shader.PropertyToID("_OrderBuffer");
            public static readonly int SplatScale = Shader.PropertyToID("_SplatScale");
            public static readonly int SplatOpacityScale = Shader.PropertyToID("_SplatOpacityScale");
            public static readonly int SplatSize = Shader.PropertyToID("_SplatSize");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int SHOrder = Shader.PropertyToID("_SHOrder");
            public static readonly int DisplayIndex = Shader.PropertyToID("_DisplayIndex");
            public static readonly int DisplayChunks = Shader.PropertyToID("_DisplayChunks");
            public static readonly int GaussianSplatRT = Shader.PropertyToID("_GaussianSplatRT");
            public static readonly int SplatSortKeys = Shader.PropertyToID("_SplatSortKeys");
            public static readonly int SplatSortDistances = Shader.PropertyToID("_SplatSortDistances");
            public static readonly int SrcBuffer = Shader.PropertyToID("_SrcBuffer");
            public static readonly int DstBuffer = Shader.PropertyToID("_DstBuffer");
            public static readonly int BufferSize = Shader.PropertyToID("_BufferSize");
            public static readonly int MatrixVP = Shader.PropertyToID("_MatrixVP");
            public static readonly int MatrixMV = Shader.PropertyToID("_MatrixMV");
            public static readonly int MatrixP = Shader.PropertyToID("_MatrixP");
            public static readonly int MatrixObjectToWorld = Shader.PropertyToID("_MatrixObjectToWorld");
            public static readonly int MatrixWorldToObject = Shader.PropertyToID("_MatrixWorldToObject");
            public static readonly int VecScreenParams = Shader.PropertyToID("_VecScreenParams");
            public static readonly int VecWorldSpaceCameraPos = Shader.PropertyToID("_VecWorldSpaceCameraPos");
            public static readonly int SelectionCenter = Shader.PropertyToID("_SelectionCenter");
            public static readonly int SelectionDelta = Shader.PropertyToID("_SelectionDelta");
            public static readonly int SelectionDeltaRotQuat = Shader.PropertyToID("_SelectionDeltaRotQuat");
            public static readonly int SelectionDeltaRotMat = Shader.PropertyToID("_SelectionDeltaRotMat");
            public static readonly int SplatCutoutsCount = Shader.PropertyToID("_SplatCutoutsCount");
            public static readonly int SplatCutouts = Shader.PropertyToID("_SplatCutouts");
            public static readonly int SplatPosMouseDown = Shader.PropertyToID("_SplatPosMouseDown");
            public static readonly int SplatOtherMouseDown = Shader.PropertyToID("_SplatOtherMouseDown");
        }

        [field: NonSerialized] public bool editModified { get; private set; }
        [field: NonSerialized] public uint editSelectedSplats { get; private set; }
        [field: NonSerialized] public uint editDeletedSplats { get; private set; }
        [field: NonSerialized] public uint editCutSplats { get; private set; }
        [field: NonSerialized] public Bounds editSelectedBounds { get; private set; }

        public GaussianSplatAsset asset => m_Asset;
        public int splatCount => m_SplatCount;

        enum KernelIndices
        {
            SetIndices,
            CalcDistances,
            CalcViewData,
            UpdateEditData,
            InitEditData,
            ClearBuffer,
            InvertSelection,
            SelectAll,
            OrBuffers,
            SelectionUpdate,
            TranslateSelection,
            RotateSelection,
            ScaleSelection,
            DuplicateSelection,
            DuplicateSelectionUpdate,
            ExportData,
            CopySplats,
        }

        public bool HasValidAsset =>
            m_Asset != null &&
            m_Asset.splatCount > 0 &&
            m_Asset.formatVersion == GaussianSplatAsset.kCurrentVersion &&
            m_Asset.posData != null &&
            m_Asset.otherData != null &&
            m_Asset.shData != null &&
            m_Asset.colorData != null;
        public bool HasValidRenderSetup => m_GpuPosData != null && m_GpuOtherData != null && m_GpuChunks != null;

        const int kGpuViewDataSize = 40;
        void CreateResourcesForAsset()
        {
            if (!HasValidAsset)
                return;

            m_SplatCount = asset.splatCount;
            m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) ((asset.posData.dataSize) / 4), 4) { name = "GaussianPosData" };
            m_GpuPosData.SetData(asset.posData.GetData<uint>());
            m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) ((asset.otherData.dataSize) / 4), 4) { name = "GaussianOtherData" };
            m_GpuOtherData.SetData(asset.otherData.GetData<uint>());
            m_GpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int) ((asset.shData.dataSize) / 4), 4) { name = "GaussianSHData" };
            m_GpuSHData.SetData(asset.shData.GetData<uint>());

            // duplicate edit tools, for storing selected splats data
            if (asset.colorFormat == GaussianSplatAsset.ColorFormat.Float32x4)
            {
                // data formats are fully lossless for duplicate edit tools
                m_GpuColorData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int)((asset.colorData.dataSize) / 4), 4) { name = "GaussianColorData" };
                m_GpuColorData.SetData(asset.colorData.GetData<uint>());
                // just a white texture, for compatibility with texture format of color data
                var tex = new Texture2D(1, 1, GraphicsFormat.R8G8B8A8_UNorm, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
                byte[] whiteColor = { 0, 0, 0, 0 };
                tex.SetPixelData(whiteColor, 0);
                tex.Apply(false, true);
                m_GpuTexColorData = tex;
            }
            else
            {
                // just a white color, for compatibility with buffer format of color data
                m_GpuColorData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 4, 4) { name = "GaussianColorData" };
                uint[] whiteColor = { 0, 0, 0, 0 };
                m_GpuColorData.SetData(whiteColor);

                var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
                var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
                var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
                tex.SetPixelData(asset.colorData.GetData<byte>(), 0);
                tex.Apply(false, true);
                m_GpuTexColorData = tex;
            }
            if (asset.chunkData != null && asset.chunkData.dataSize != 0)
            {
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    (int) (asset.chunkData.dataSize / UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()),
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunks.SetData(asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());
                m_GpuChunksValid = true;
            }
            else
            {
                // just a dummy chunk buffer
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1,
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunksValid = false;
            }

            m_GpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_Asset.splatCount, kGpuViewDataSize);
            m_GpuIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 36, 2);
            // cube indices, most often we use only the first quad
            m_GpuIndexBuffer.SetData(new ushort[]
            {
                0, 1, 2, 1, 3, 2,
                4, 6, 5, 5, 6, 7,
                0, 2, 4, 4, 2, 6,
                1, 5, 3, 5, 7, 3,
                0, 4, 1, 4, 5, 1,
                2, 3, 6, 3, 7, 6
            });

            InitSortBuffers(m_Asset.splatCount);
        }

        void CreateBufferForDuplicate(int selectedSplatCount)
        {
            //alloc extra buffers for storing duplicated splats's data
            int dupSplatCount = selectedSplatCount;
            if (dupSplatCount != 0 && dupSplatCount != preDupSplatCount)
            {
                // need to recreate buffers for duplicate
                DisposeBuffer(ref m_GpuDupPosData);
                DisposeBuffer(ref m_GpuDupOtherData);
                DisposeBuffer(ref m_GpuDupSHData);
                DisposeBuffer(ref m_GpuDupColorData);

                // calc  splat buffers's size for the redundant buffers
                long dupGpuPosDataSize = GaussianSplatAsset.CalcPosDataSize(dupSplatCount, asset.posFormat);
                long dupGpuOtherDataSize = GaussianSplatAsset.CalcOtherDataSize(dupSplatCount, asset.scaleFormat);
                long dupGpuSHDataSize = GaussianSplatAsset.CalcSHDataSize(dupSplatCount, asset.shFormat);
                long dupGpuColorDataSize = GaussianSplatAsset.CalcColorDataSize(dupSplatCount, asset.colorFormat);

                if (m_GpuDupPosData == null)
                {
                    m_GpuDupPosData = new GraphicsBuffer(GraphicsBuffer.Target.Append, (int)(dupGpuPosDataSize), (int)(UnsafeUtility.SizeOf<float3>())) { name = "GaussianDuplicatedPosData" };
                }
                if (m_GpuDupOtherData == null)
                {
                    m_GpuDupOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Append, (int)(dupGpuOtherDataSize), (int)(UnsafeUtility.SizeOf<float4>())) { name = "GaussianDuplicatedOtherData" };
                }
                if (m_GpuDupSHData == null)
                {
                    m_GpuDupSHData = new GraphicsBuffer(GraphicsBuffer.Target.Append, (int)(dupGpuSHDataSize), (int)(UnsafeUtility.SizeOf<GaussianSplatAsset.SHTableItemFloat32>())) { name = "GaussianDuplicatedSHData" };
                }
                if (m_GpuDupColorData == null)
                {
                    m_GpuDupColorData = new GraphicsBuffer(GraphicsBuffer.Target.Append, (int)(dupGpuColorDataSize), (int)(UnsafeUtility.SizeOf<float4>())) { name = "GaussianDuplicatedColorData" };
                }
                preDupSplatCount = dupSplatCount;
            }
        }

        void ReCreateResourcesForDuplicate(int splatCount, NativeArray<byte> posData, NativeArray<byte> otherData, NativeArray<byte> shData, NativeArray<byte> colorData)
        {
            if (!posData.IsCreated
                || !otherData.IsCreated
                || !shData.IsCreated
                || !colorData.IsCreated)
                return;

            DisposeResourcesForDuplicate();

            m_SplatCount = splatCount;
            m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int)((posData.Length) / 4), 4) { name = "GaussianPosData" };
            m_GpuPosData.SetData(posData.Reinterpret<uint>(1));
            m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int)((otherData.Length) / 4), 4) { name = "GaussianOtherData" };
            m_GpuOtherData.SetData(otherData.Reinterpret<uint>(1));
            m_GpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int)((shData.Length) / 4), 4) { name = "GaussianSHData" };
            m_GpuSHData.SetData(shData.Reinterpret<uint>(1));

            Assert.IsTrue(asset.colorFormat == GaussianSplatAsset.ColorFormat.Float32x4);

            // duplicate edit tools, for storing selected splats data
            // data formats are fully lossless for duplicate edit tools
            m_GpuColorData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int)((colorData.Length) / 4), 4) { name = "GaussianColorData" };
            m_GpuColorData.SetData(colorData.Reinterpret<uint>(1));
            // just a white texture
            var tex = new Texture2D(1, 1, GraphicsFormat.R8G8B8A8_UNorm, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
            byte[] whiteColor = { 0, 0, 0, 0 };
            tex.SetPixelData(whiteColor, 0);
            tex.Apply(false, true);
            m_GpuTexColorData = tex;

            m_GpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, splatCount, kGpuViewDataSize);
            InitSortBuffers(splatCount);

        }
        void InitSortBuffers(int splatCount) 
        {
            m_GpuSortDistances?.Dispose();
            m_GpuSortKeys?.Dispose();
            m_SorterArgs.resources.Dispose();

            m_GpuSortDistances = new GraphicsBuffer(GraphicsBuffer.Target.Structured, splatCount, 4) { name = "GaussianSplatSortDistances" };
            m_GpuSortKeys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, splatCount, 4) { name = "GaussianSplatSortIndices" };

            // init keys buffer to splat indices
            m_CSSplatUtilities.SetBuffer((int)KernelIndices.SetIndices, Props.SplatSortKeys, m_GpuSortKeys);
            m_CSSplatUtilities.SetInt(Props.SplatCount, m_GpuSortDistances.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.SetIndices, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch((int)KernelIndices.SetIndices, (m_GpuSortDistances.count + (int)gsX - 1) / (int)gsX, 1, 1);

            m_SorterArgs.inputKeys = m_GpuSortDistances;
            m_SorterArgs.inputValues = m_GpuSortKeys;
            m_SorterArgs.count = (uint)splatCount;
            if (m_Sorter.Valid)
                m_SorterArgs.resources = GpuSorting.SupportResources.Load((uint)splatCount);
        }
        public void OnEnable()
        {
            m_FrameCounter = 0;
            if (m_ShaderSplats == null || m_ShaderComposite == null || m_ShaderDebugPoints == null || m_ShaderDebugBoxes == null || m_CSSplatUtilities == null)
                return;
            if (!SystemInfo.supportsComputeShaders)
                return;

            m_MatSplats = new Material(m_ShaderSplats) {name = "GaussianSplats"};
            m_MatComposite = new Material(m_ShaderComposite) {name = "GaussianClearDstAlpha"};
            m_MatDebugPoints = new Material(m_ShaderDebugPoints) {name = "GaussianDebugPoints"};
            m_MatDebugBoxes = new Material(m_ShaderDebugBoxes) {name = "GaussianDebugBoxes"};

            m_Sorter = new GpuSorting(m_CSSplatUtilities);
            GaussianSplatRenderSystem.instance.RegisterSplat(this);

            CreateResourcesForAsset();
        }

        void SetAssetDataOnCS(CommandBuffer cmb, KernelIndices kernel)
        {
            ComputeShader cs = m_CSSplatUtilities;
            int kernelIndex = (int) kernel;
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatPos, m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatChunks, m_GpuChunks);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatOther, m_GpuOtherData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSH, m_GpuSHData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatColor, m_GpuColorData);
            cmb.SetComputeTextureParam(cs, kernelIndex, Props.SplatTexColor, m_GpuTexColorData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSelectedBits, m_GpuEditSelected ?? m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatDeletedBits, m_GpuEditDeleted ?? m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatViewData, m_GpuView);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.OrderBuffer, m_GpuSortKeys);

            cmb.SetComputeIntParam(cs, Props.SplatBitsValid, m_GpuEditSelected != null && m_GpuEditDeleted != null ? 1 : 0);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16) | ((uint)m_Asset.colorFormat << 24);
            cmb.SetComputeIntParam(cs, Props.SplatFormat, (int)format);
            cmb.SetComputeIntParam(cs, Props.SplatCount, m_SplatCount);
            cmb.SetComputeIntParam(cs, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);

            UpdateCutoutsBuffer();
            cmb.SetComputeIntParam(cs, Props.SplatCutoutsCount, m_Cutouts?.Length ?? 0);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatCutouts, m_GpuEditCutouts);
        }

        internal void SetAssetDataOnMaterial(MaterialPropertyBlock mat)
        {
            mat.SetBuffer(Props.SplatPos, m_GpuPosData);
            mat.SetBuffer(Props.SplatOther, m_GpuOtherData);
            mat.SetBuffer(Props.SplatSH, m_GpuSHData);
            mat.SetBuffer(Props.SplatColor, m_GpuColorData);
            mat.SetTexture(Props.SplatTexColor, m_GpuTexColorData);
            mat.SetBuffer(Props.SplatSelectedBits, m_GpuEditSelected ?? m_GpuPosData);
            mat.SetBuffer(Props.SplatDeletedBits, m_GpuEditDeleted ?? m_GpuPosData);
            mat.SetInt(Props.SplatBitsValid, m_GpuEditSelected != null && m_GpuEditDeleted != null ? 1 : 0);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            mat.SetInteger(Props.SplatFormat, (int)format);
            mat.SetInteger(Props.SplatCount, m_SplatCount);
            mat.SetInteger(Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
        }

        static void DisposeBuffer(ref GraphicsBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }

        void DisposeResourcesForAsset()
        {
            DestroyImmediate(m_GpuTexColorData);
            DisposeBuffer(ref m_GpuColorData);

            DisposeBuffer(ref m_GpuDupPosData);
            DisposeBuffer(ref m_GpuDupOtherData);
            DisposeBuffer(ref m_GpuDupSHData);
            DisposeBuffer(ref m_GpuDupColorData);

            DisposeBuffer(ref m_GpuPosData);
            DisposeBuffer(ref m_GpuOtherData);
            DisposeBuffer(ref m_GpuSHData);

            DisposeBuffer(ref m_GpuView);
            DisposeBuffer(ref m_GpuIndexBuffer);
            DisposeBuffer(ref m_GpuSortDistances);
            DisposeBuffer(ref m_GpuSortKeys);

            DisposeBuffer(ref m_GpuChunks);

            DisposeBuffer(ref m_GpuEditSelectedMouseDown);
            DisposeBuffer(ref m_GpuEditPosMouseDown);
            DisposeBuffer(ref m_GpuEditOtherMouseDown);
            DisposeBuffer(ref m_GpuEditSelected);
            DisposeBuffer(ref m_GpuEditDeleted);
            DisposeBuffer(ref m_GpuEditCountsBounds);
            DisposeBuffer(ref m_GpuEditCutouts);

            m_SorterArgs.resources.Dispose();

            m_SplatCount = 0;
            m_GpuChunksValid = false;

            editSelectedSplats = 0;
            editDeletedSplats = 0;
            editCutSplats = 0;
            editModified = false;
            editSelectedBounds = default;
        }

        void DisposeResourcesForDuplicate()
        {
            DestroyImmediate(m_GpuTexColorData);
            DisposeBuffer(ref m_GpuColorData);

            DisposeBuffer(ref m_GpuPosData);
            DisposeBuffer(ref m_GpuOtherData);
            DisposeBuffer(ref m_GpuSHData);

            DisposeBuffer(ref m_GpuView);
            DisposeBuffer(ref m_GpuSortDistances);
            DisposeBuffer(ref m_GpuSortKeys);

            DisposeBuffer(ref m_GpuEditSelectedMouseDown);
            DisposeBuffer(ref m_GpuEditPosMouseDown);
            DisposeBuffer(ref m_GpuEditOtherMouseDown);
            DisposeBuffer(ref m_GpuEditSelected);
            DisposeBuffer(ref m_GpuEditDeleted);

            m_SorterArgs.resources.Dispose();

            m_SplatCount = 0;
        }

        public void OnDisable()
        {
            DisposeResourcesForAsset();
            GaussianSplatRenderSystem.instance.UnregisterSplat(this);

            DestroyImmediate(m_MatSplats);
            DestroyImmediate(m_MatComposite);
            DestroyImmediate(m_MatDebugPoints);
            DestroyImmediate(m_MatDebugBoxes);
        }

        internal void CalcViewData(CommandBuffer cmb, Camera cam, Matrix4x4 matrix)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            var tr = transform;

            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matProj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            Vector4 screenPar = new Vector4(screenW, screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            // calculate view dependent data for each splat
            SetAssetDataOnCS(cmb, KernelIndices.CalcViewData);

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixVP, matProj * matView);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixP, matProj);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatScale, m_SplatScale);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatOpacityScale, m_OpacityScale);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOrder, m_SHOrder);

            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcViewData, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcViewData, (m_GpuView.count + (int)gsX - 1)/(int)gsX, 1, 1);
        }

        internal void SortPoints(CommandBuffer cmd, Camera cam, Matrix4x4 matrix)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            Matrix4x4 worldToCamMatrix = cam.worldToCameraMatrix;
            worldToCamMatrix.m20 *= -1;
            worldToCamMatrix.m21 *= -1;
            worldToCamMatrix.m22 *= -1;

            // calculate distance to the camera for each splat
            cmd.BeginSample(s_ProfSort);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatSortDistances, m_GpuSortDistances);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatSortKeys, m_GpuSortKeys);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatChunks, m_GpuChunks);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatPos, m_GpuPosData);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatFormat, (int)m_Asset.posFormat);
            cmd.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, worldToCamMatrix * matrix);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatCount, m_SplatCount);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcDistances, out uint gsX, out _, out _);
            cmd.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, (m_GpuSortDistances.count + (int)gsX - 1)/(int)gsX, 1, 1);

            // sort the splats
            m_Sorter.Dispatch(cmd, m_SorterArgs);
            cmd.EndSample(s_ProfSort);
        }

        public void Update()
        {
            var curHash = m_Asset ? m_Asset.dataHash : new Hash128();
            if (m_PrevAsset != m_Asset || m_PrevHash != curHash)
            {
                m_PrevAsset = m_Asset;
                m_PrevHash = curHash;
                DisposeResourcesForAsset();
                CreateResourcesForAsset();
            }
        }

        public void ActivateCamera(int index)
        {
            Camera mainCam = Camera.main;
            if (!mainCam)
                return;
            if (!m_Asset || m_Asset.cameras == null)
                return;

            var selfTr = transform;
            var camTr = mainCam.transform;
            var prevParent = camTr.parent;
            var cam = m_Asset.cameras[index];
            camTr.parent = selfTr;
            camTr.localPosition = cam.pos;
            camTr.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
            camTr.parent = prevParent;
            camTr.localScale = Vector3.one;
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(camTr);
#endif
        }

        void ClearGraphicsBuffer(GraphicsBuffer buf)
        {
            m_CSSplatUtilities.SetBuffer((int)KernelIndices.ClearBuffer, Props.DstBuffer, buf);
            m_CSSplatUtilities.SetInt(Props.BufferSize, buf.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.ClearBuffer, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch((int)KernelIndices.ClearBuffer, (int)((buf.count+gsX-1)/gsX), 1, 1);
        }

        void UnionGraphicsBuffers(GraphicsBuffer dst, GraphicsBuffer src)
        {
            m_CSSplatUtilities.SetBuffer((int)KernelIndices.OrBuffers, Props.SrcBuffer, src);
            m_CSSplatUtilities.SetBuffer((int)KernelIndices.OrBuffers, Props.DstBuffer, dst);
            m_CSSplatUtilities.SetInt(Props.BufferSize, dst.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.OrBuffers, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch((int)KernelIndices.OrBuffers, (int)((dst.count+gsX-1)/gsX), 1, 1);
        }

        static float SortableUintToFloat(uint v)
        {
            uint mask = ((v >> 31) - 1) | 0x80000000u;
            return math.asfloat(v ^ mask);
        }

        public void UpdateEditCountsAndBounds()
        {
            if (m_GpuEditSelected == null)
            {
                editSelectedSplats = 0;
                editDeletedSplats = 0;
                editCutSplats = 0;
                editModified = false;
                editSelectedBounds = default;
                return;
            }

            m_CSSplatUtilities.SetBuffer((int)KernelIndices.InitEditData, Props.DstBuffer, m_GpuEditCountsBounds);
            m_CSSplatUtilities.Dispatch((int)KernelIndices.InitEditData, 1, 1, 1);

            using CommandBuffer cmb = new CommandBuffer();
            SetAssetDataOnCS(cmb, KernelIndices.UpdateEditData);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.UpdateEditData, Props.DstBuffer, m_GpuEditCountsBounds);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.BufferSize, m_GpuEditSelected.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.UpdateEditData, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.UpdateEditData, (int)((m_GpuEditSelected.count+gsX-1)/gsX), 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);

            uint[] res = new uint[m_GpuEditCountsBounds.count];
            m_GpuEditCountsBounds.GetData(res);
            editSelectedSplats = res[0];
            editDeletedSplats = res[1];
            editCutSplats = res[2];
            Vector3 min = new Vector3(SortableUintToFloat(res[3]), SortableUintToFloat(res[4]), SortableUintToFloat(res[5]));
            Vector3 max = new Vector3(SortableUintToFloat(res[6]), SortableUintToFloat(res[7]), SortableUintToFloat(res[8]));
            Bounds bounds = default;
            bounds.SetMinMax(min, max);
            if (bounds.extents.sqrMagnitude < 0.01)
                bounds.extents = new Vector3(0.1f,0.1f,0.1f);
            editSelectedBounds = bounds;
        }

        void UpdateCutoutsBuffer()
        {
            int bufferSize = m_Cutouts?.Length ?? 0;
            if (bufferSize == 0)
                bufferSize = 1;
            if (m_GpuEditCutouts == null || m_GpuEditCutouts.count != bufferSize)
            {
                m_GpuEditCutouts?.Dispose();
                m_GpuEditCutouts = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bufferSize, UnsafeUtility.SizeOf<GaussianCutout.ShaderData>()) { name = "GaussianCutouts" };
            }

            NativeArray<GaussianCutout.ShaderData> data = new(bufferSize, Allocator.Temp);
            if (m_Cutouts != null)
            {
                var matrix = transform.localToWorldMatrix;
                for (var i = 0; i < m_Cutouts.Length; ++i)
                {
                    data[i] = GaussianCutout.GetShaderData(m_Cutouts[i], matrix);
                }
            }

            m_GpuEditCutouts.SetData(data);
            data.Dispose();
        }

        bool EnsureEditingBuffers()
        {
            if (!HasValidAsset || !HasValidRenderSetup)
                return false;

            if (m_GpuEditSelected == null)
            {
                var target = GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource |
                             GraphicsBuffer.Target.CopyDestination;
                var size = (m_SplatCount + 31) / 32;
                m_GpuEditSelected = new GraphicsBuffer(target, size, 4) {name = "GaussianSplatSelected"};
                m_GpuEditSelectedMouseDown = new GraphicsBuffer(target, size, 4) {name = "GaussianSplatSelectedInit"};
                m_GpuEditDeleted = new GraphicsBuffer(target, size, 4) {name = "GaussianSplatDeleted"};
                m_GpuEditCountsBounds = new GraphicsBuffer(target, 3 + 6, 4) {name = "GaussianSplatEditData"}; // selected count, deleted bound, cut count, float3 min, float3 max
                ClearGraphicsBuffer(m_GpuEditSelected);
                ClearGraphicsBuffer(m_GpuEditSelectedMouseDown);
                ClearGraphicsBuffer(m_GpuEditDeleted);
            }
            return m_GpuEditSelected != null;
        }

        public void EditStoreSelectionMouseDown()
        {
            if (!EnsureEditingBuffers()) return;
            Graphics.CopyBuffer(m_GpuEditSelected, m_GpuEditSelectedMouseDown);
        }

        public void EditStorePosMouseDown()
        {
            if (m_GpuEditPosMouseDown == null)
            {
                m_GpuEditPosMouseDown = new GraphicsBuffer(m_GpuPosData.target | GraphicsBuffer.Target.CopyDestination, m_GpuPosData.count, m_GpuPosData.stride) {name = "GaussianSplatEditPosMouseDown"};
            }
            Graphics.CopyBuffer(m_GpuPosData, m_GpuEditPosMouseDown);
        }
        public void EditStoreOtherMouseDown()
        {
            if (m_GpuEditOtherMouseDown == null)
            {
                m_GpuEditOtherMouseDown = new GraphicsBuffer(m_GpuOtherData.target | GraphicsBuffer.Target.CopyDestination, m_GpuOtherData.count, m_GpuOtherData.stride) {name = "GaussianSplatEditOtherMouseDown"};
            }
            Graphics.CopyBuffer(m_GpuOtherData, m_GpuEditOtherMouseDown);
        }

        public void EditUpdateSelection(Vector2 rectMin, Vector2 rectMax, Camera cam, bool subtract)
        {
            if (!EnsureEditingBuffers()) return;

            Graphics.CopyBuffer(m_GpuEditSelectedMouseDown, m_GpuEditSelected);

            var tr = transform;
            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matProj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            Vector4 screenPar = new Vector4(screenW, screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            using var cmb = new CommandBuffer { name = "SplatSelectionUpdate" };
            SetAssetDataOnCS(cmb, KernelIndices.SelectionUpdate);

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixVP, matProj * matView);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixP, matProj);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_SelectionRect", new Vector4(rectMin.x, rectMax.y, rectMax.x, rectMin.y));
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SelectionMode, subtract ? 0 : 1);

            DispatchUtilsAndExecute(cmb, KernelIndices.SelectionUpdate, m_SplatCount);
            UpdateEditCountsAndBounds();
        }

        public void EditTranslateSelection(Vector3 localSpacePosDelta)
        {
            if (!EnsureEditingBuffers()) return;

            using var cmb = new CommandBuffer { name = "SplatTranslateSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.TranslateSelection);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionDelta, localSpacePosDelta);

            DispatchUtilsAndExecute(cmb, KernelIndices.TranslateSelection, m_SplatCount);
            UpdateEditCountsAndBounds();
            editModified = true;
        }

        public void EditRotateSelection(Vector3 localSpaceCenter, Matrix4x4 localToWorld, Matrix4x4 worldToLocal, Quaternion rotation)
        {
            if (!EnsureEditingBuffers()) return;
            if (m_GpuEditPosMouseDown == null || m_GpuEditOtherMouseDown == null) return; // should have captured initial state

            using var cmb = new CommandBuffer { name = "SplatRotateSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.RotateSelection);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RotateSelection, Props.SplatPosMouseDown, m_GpuEditPosMouseDown);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RotateSelection, Props.SplatOtherMouseDown, m_GpuEditOtherMouseDown);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionCenter, localSpaceCenter);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, localToWorld);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, worldToLocal);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionDeltaRotQuat, new Vector4(rotation.x, rotation.y, rotation.z, rotation.w));
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.SelectionDeltaRotMat, Matrix4x4.Rotate(rotation));
            DispatchUtilsAndExecute(cmb, KernelIndices.RotateSelection, m_SplatCount);
            UpdateEditCountsAndBounds();
            editModified = true;
        }


        public void EditScaleSelection(Vector3 localSpaceCenter, Matrix4x4 localToWorld, Matrix4x4 worldToLocal, Vector3 scale)
        {
            if (!EnsureEditingBuffers()) return;
            if (m_GpuEditPosMouseDown == null || m_GpuEditOtherMouseDown == null) return; // should have captured initial state

            using var cmb = new CommandBuffer { name = "SplatScaleSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.ScaleSelection);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ScaleSelection, Props.SplatPosMouseDown, m_GpuEditPosMouseDown);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ScaleSelection, Props.SplatOtherMouseDown, m_GpuEditOtherMouseDown);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionCenter, localSpaceCenter);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, localToWorld);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, worldToLocal);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionDelta, scale);

            DispatchUtilsAndExecute(cmb, KernelIndices.ScaleSelection, m_SplatCount);
            UpdateEditCountsAndBounds();
            editModified = true;
        }

        public void EditDeleteSelected()
        {
            if (!EnsureEditingBuffers()) return;
            UnionGraphicsBuffers(m_GpuEditDeleted, m_GpuEditSelected);
            EditDeselectAll();
            UpdateEditCountsAndBounds();
            if (editDeletedSplats != 0)
                editModified = true;
        }

        public void EditSelectAll()
        {
            if (!EnsureEditingBuffers()) return;
            using var cmb = new CommandBuffer { name = "SplatSelectAll" };
            SetAssetDataOnCS(cmb, KernelIndices.SelectAll);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.SelectAll, Props.DstBuffer, m_GpuEditSelected);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.BufferSize, m_GpuEditSelected.count);
            DispatchUtilsAndExecute(cmb, KernelIndices.SelectAll, m_GpuEditSelected.count);
            UpdateEditCountsAndBounds();
        }

        public void EditDeselectAll()
        {
            if (!EnsureEditingBuffers()) return;
            ClearGraphicsBuffer(m_GpuEditSelected);
            UpdateEditCountsAndBounds();
        }

        public void EditInvertSelection()
        {
            if (!EnsureEditingBuffers()) return;

            using var cmb = new CommandBuffer { name = "SplatInvertSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.InvertSelection);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InvertSelection, Props.DstBuffer, m_GpuEditSelected);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.BufferSize, m_GpuEditSelected.count);
            DispatchUtilsAndExecute(cmb, KernelIndices.InvertSelection, m_GpuEditSelected.count);
            UpdateEditCountsAndBounds();
        }

        public uint GetAppendStructuredBufferCount(ref GraphicsBuffer src)
        {
            uint[] count = new uint[1] { 0 };
            var computeBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.IndirectArguments);
            GraphicsBuffer.CopyCount(src, computeBuffer, 0);
            computeBuffer.GetData(count);
            computeBuffer.Dispose();
            return count[0];
        }

        struct MergeArraysJob<T> : IJobParallelFor where T : struct

        {
            [ReadOnly] public NativeArray<T> array1;
            [ReadOnly] public NativeArray<T> array2;
            public NativeArray<T> mergedArray;

            public void Execute(int index)
            {
                if (index < array1.Length)
                {
                    mergedArray[index] = array1[index];
                }
                else
                {
                    mergedArray[index] = array2[index - array1.Length];
                }
            }
        }

        public void EditDuplicateSelection()
        {
            if (asset.colorFormat > GaussianSplatAsset.ColorFormat.Float32x4)
            {
                Debug.LogWarning("It is recommended to use High or VeryHigh quality preset for editing splats, lower levels are lossy. And copy nothing this time.");
                return;
            }
            if (!EnsureEditingBuffers()) return;

            int selectedSplatCount = (int)editSelectedSplats;
 
            if (selectedSplatCount <= 0) return;

            //CreateBufferForDuplicate(selectedSplatCount);

            //using var cmb = new CommandBuffer { name = "SplatDuplicateSelection" };
            //SetAssetDataOnCS(cmb, KernelIndices.DuplicateSelection);

            // reset buffer counter
            //m_GpuDupPosData.SetCounterValue(0);
            //m_GpuDupOtherData.SetCounterValue(0);
            //m_GpuDupSHData.SetCounterValue(0);
            //m_GpuDupColorData.SetCounterValue(0);

            //cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateSelection, Props.SplatDuplicatedPos, m_GpuDupPosData);
            //cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateSelection, Props.SplatDuplicatedOther, m_GpuDupOtherData);
            //cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateSelection, Props.SplatDuplicatedSH, m_GpuDupSHData);
            //cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateSelection, Props.SplatDuplicatedColor, m_GpuDupColorData);

            //DispatchUtilsAndExecute(cmb, KernelIndices.DuplicateSelection, m_SplatCount);

            //uint posDupCount = GetAppendStructuredBufferCount(ref m_GpuDupPosData);
            //uint otherDupCount = GetAppendStructuredBufferCount(ref m_GpuDupOtherData);
            //uint shDupCount = GetAppendStructuredBufferCount(ref m_GpuDupSHData);
            //uint colorDupCount = GetAppendStructuredBufferCount(ref m_GpuDupColorData);

            //// just check whether selectedSplatCount is equal to pos\sh\other\color dup count or not
            //Assert.IsTrue((selectedSplatCount == posDupCount) && (posDupCount == shDupCount) && (posDupCount == otherDupCount) && (posDupCount == colorDupCount));

            int dupSplatCount = selectedSplatCount;
  
            float3[] posDupArray= new float3[dupSplatCount];
            float4[] otherDupArray = new float4[dupSplatCount];
            float4[] colorDupArray = new float4[dupSplatCount];
            GaussianSplatAsset.SHTableItemFloat32[] shDupArray = new GaussianSplatAsset.SHTableItemFloat32[dupSplatCount];
            // get duplicated splat data from gpu memory
            //m_GpuDupPosData.GetData(posDupArray);
            //m_GpuDupOtherData.GetData(otherDupArray);
            //m_GpuDupSHData.GetData(shDupArray);
            //m_GpuDupColorData.GetData(colorDupArray);

            float3[] posPreArray = new float3[m_SplatCount];
            float4[] otherPreArray = new float4[m_SplatCount];
            float4[] colorPreArray = new float4[m_SplatCount];
            GaussianSplatAsset.SHTableItemFloat32[] shPreArray = new GaussianSplatAsset.SHTableItemFloat32[m_SplatCount];
            // get previous splat data from gpu memory
            m_GpuPosData.GetData(posPreArray);
            m_GpuOtherData.GetData(otherPreArray);
            m_GpuSHData.GetData(shPreArray);
            m_GpuColorData.GetData(colorPreArray);

            var gpuSelected = m_GpuEditSelected;
            uint[] selected = new uint[gpuSelected.count];
            gpuSelected.GetData(selected);

            int selectedID = 0;
            for (int i = 0; i < m_SplatCount; ++i)
            {
                int wordIdx = i >> 5;
                int bitIdx = i & 31;
                bool isSelected = (selected[wordIdx] & (1u << bitIdx)) != 0;
                if (isSelected) {
                    posDupArray[selectedID] = posPreArray[i];
                    otherDupArray[selectedID] = otherPreArray[i];
                    colorDupArray[selectedID] = colorPreArray[i];
                    shDupArray[selectedID] = shPreArray[i];
                    selectedID++;
                }
            }

            Assert.IsTrue(selectedID == dupSplatCount);

            float3[] posMergedArray = new float3[posPreArray.Length + posDupArray.Length];
            float4[] otherMergedArray = new float4[otherPreArray.Length + otherDupArray.Length];
            float4[] colorMergedArray = new float4[colorPreArray.Length + colorDupArray.Length];
            GaussianSplatAsset.SHTableItemFloat32[] shMergedArray = new GaussianSplatAsset.SHTableItemFloat32[shPreArray.Length + shDupArray.Length];

            NativeArray<float3> posData = new NativeArray<float3>(posPreArray.Length + posDupArray.Length, Allocator.TempJob);
            NativeArray<float4> otherData = new NativeArray<float4>(otherPreArray.Length + otherDupArray.Length, Allocator.TempJob);
            NativeArray<float4> colorData = new NativeArray<float4>(colorPreArray.Length + colorDupArray.Length, Allocator.TempJob);
            NativeArray<GaussianSplatAsset.SHTableItemFloat32> shData = new NativeArray<GaussianSplatAsset.SHTableItemFloat32>(shPreArray.Length + shDupArray.Length, Allocator.TempJob);
            // merge data
            MergeArraysJob<float3> posDataMergeJob = new MergeArraysJob<float3>
            {
                array1 = new NativeArray<float3>(posPreArray, Allocator.TempJob),
                array2 = new NativeArray<float3>(posDupArray, Allocator.TempJob),
                mergedArray = posData
            };
            posDataMergeJob.Schedule(posMergedArray.Length, 8192).Complete();
            posDataMergeJob.array1.Dispose();
            posDataMergeJob.array2.Dispose();

            MergeArraysJob<float4> otherDataMergeJob = new MergeArraysJob<float4>
            {
                array1 = new NativeArray<float4>(otherPreArray, Allocator.TempJob),
                array2 = new NativeArray<float4>(otherDupArray, Allocator.TempJob),
                mergedArray = otherData
            };
            otherDataMergeJob.Schedule(otherMergedArray.Length, 8192).Complete();
            otherDataMergeJob.array1.Dispose();
            otherDataMergeJob.array2.Dispose();

            MergeArraysJob<float4> colorMergeJob = new MergeArraysJob<float4>
            {
                array1 = new NativeArray<float4>(colorPreArray, Allocator.TempJob),
                array2 = new NativeArray<float4>(colorDupArray, Allocator.TempJob),
                mergedArray = colorData
            };
            colorMergeJob.Schedule(colorMergedArray.Length, 8192).Complete();
            colorMergeJob.array1.Dispose();
            colorMergeJob.array2.Dispose();

            MergeArraysJob<GaussianSplatAsset.SHTableItemFloat32> shMergeJob = new MergeArraysJob<GaussianSplatAsset.SHTableItemFloat32>
            {
                array1 = new NativeArray<GaussianSplatAsset.SHTableItemFloat32>(shPreArray, Allocator.TempJob),
                array2 = new NativeArray<GaussianSplatAsset.SHTableItemFloat32>(shDupArray, Allocator.TempJob),
                mergedArray = shData
            };
            shMergeJob.Schedule(shMergedArray.Length, 8192).Complete();
            shMergeJob.array1.Dispose();
            shMergeJob.array2.Dispose();

            //Array.Copy(posPreArray, 0, posMergedArray, 0, posPreArray.Length);
            //Array.Copy(posDupArray, 0, posMergedArray, posPreArray.Length, posDupArray.Length);

            //Array.Copy(otherPreArray, 0, otherMergedArray, 0, otherPreArray.Length);
            //Array.Copy(otherDupArray, 0, otherMergedArray, otherPreArray.Length, otherDupArray.Length);

            //Array.Copy(shPreArray, 0, shMergedArray, 0, shPreArray.Length);
            //Array.Copy(shDupArray, 0, shMergedArray, shPreArray.Length, shDupArray.Length);

            //Array.Copy(colorPreArray, 0, colorMergedArray, 0, colorPreArray.Length);
            //Array.Copy(colorDupArray, 0, colorMergedArray, colorPreArray.Length, colorDupArray.Length);

            //NativeArray<float3> posData = new NativeArray<float3>(posMergedArray, Allocator.Temp);
            //NativeArray<float4> otherData = new NativeArray<float4>(otherMergedArray, Allocator.Temp);
            //NativeArray<float4> colorData = new NativeArray<float4>(colorMergedArray, Allocator.Temp);
            //NativeArray<GaussianSplatAsset.SHTableItemFloat32> shData = new NativeArray<GaussianSplatAsset.SHTableItemFloat32>(shMergedArray, Allocator.Temp);

            int mergedSplatCount = m_SplatCount + dupSplatCount;

            // store delete bits to cpu memory
            var gpuDeleted = m_GpuEditDeleted;
            uint[] deleted = new uint[gpuDeleted.count];
            gpuDeleted.GetData(deleted);

            var mergedSize = (mergedSplatCount + 31) / 32;
            uint[] mergedDeleted = new uint[mergedSize];
            Array.Copy(deleted, 0, mergedDeleted, 0, deleted.Length);

            // since splat count is larger after duplicate operation, all gpu resources  need to realloc
            ReCreateResourcesForDuplicate(mergedSplatCount, posData.Reinterpret<byte>(UnsafeUtility.SizeOf<float3>()), otherData.Reinterpret<byte>(UnsafeUtility.SizeOf<float4>()),
                                            shData.Reinterpret<byte>(UnsafeUtility.SizeOf<GaussianSplatAsset.SHTableItemFloat32>()),
                                            colorData.Reinterpret<byte>(UnsafeUtility.SizeOf<float4>()));
            posData.Dispose();
            otherData.Dispose();
            colorData.Dispose();
            shData.Dispose();

            // since splat count is larger after duplicate operation, all gpu resources  need to realloc
            if (m_GpuEditSelected == null)
            {
                var target = GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource |
                             GraphicsBuffer.Target.CopyDestination;
                var size = (m_SplatCount + 31) / 32;
                m_GpuEditSelected = new GraphicsBuffer(target, size, 4) { name = "GaussianSplatSelected" };
                m_GpuEditSelectedMouseDown = new GraphicsBuffer(target, size, 4) { name = "GaussianSplatSelectedInit" };
                m_GpuEditDeleted = new GraphicsBuffer(target, size, 4) { name = "GaussianSplatDeleted" };
                ClearGraphicsBuffer(m_GpuEditSelected);
                ClearGraphicsBuffer(m_GpuEditSelectedMouseDown);
                ClearGraphicsBuffer(m_GpuEditDeleted);
            }

            // restore delete bits tp gpu memory before duplicate operation
            m_GpuEditDeleted.SetData(mergedDeleted);

            // restore selection bits before duplicate operation, but set selection bits on the duplicated splats
            using var cmb2 = new CommandBuffer { name = "SplatDuplicateSelectionUpdate" };
            SetAssetDataOnCS(cmb2, KernelIndices.DuplicateSelectionUpdate);
            cmb2.SetComputeIntParam(m_CSSplatUtilities, Props.DupSplatCount, dupSplatCount);
            DispatchUtilsAndExecute(cmb2, KernelIndices.DuplicateSelectionUpdate, m_SplatCount);

            editModified = true;

            Debug.Log($"Duplicate {dupSplatCount} splats success!");
            UpdateEditCountsAndBounds();
        }

        public bool EditExportData(GraphicsBuffer dstData, bool bakeTransform)
        {
            if (!EnsureEditingBuffers()) return false;

            int flags = 0;
            var tr = transform;
            Quaternion bakeRot = tr.localRotation;
            Vector3 bakeScale = tr.localScale;

            if (bakeTransform)
                flags = 1;

            using var cmb = new CommandBuffer { name = "SplatExportData" };
            SetAssetDataOnCS(cmb, KernelIndices.ExportData);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_ExportTransformFlags", flags);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_ExportTransformRotation", new Vector4(bakeRot.x, bakeRot.y, bakeRot.z, bakeRot.w));
            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_ExportTransformScale", bakeScale);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, tr.localToWorldMatrix);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ExportData, "_ExportBuffer", dstData);

            DispatchUtilsAndExecute(cmb, KernelIndices.ExportData, m_SplatCount);
            return true;
        }

        public void EditSetSplatCount(int newSplatCount)
        {
            if (newSplatCount <= 0 || newSplatCount > GaussianSplatAsset.kMaxSplats)
            {
                Debug.LogError($"Invalid new splat count: {newSplatCount}");
                return;
            }
            if (asset.chunkData != null)
            {
                Debug.LogError("Only splats with VeryHigh quality can be resized");
                return;
            }
            if (newSplatCount == splatCount)
                return;

            int posStride = (int)(asset.posData.dataSize / asset.splatCount);
            int otherStride = (int)(asset.otherData.dataSize / asset.splatCount);
            int shStride = (int)(asset.shData.dataSize / asset.splatCount);
            int colorStride = (int)(asset.colorData.dataSize / asset.splatCount);

            // create new GPU buffers
            var newPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, newSplatCount * posStride / 4, 4) { name = "GaussianPosData" };
            var newOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, newSplatCount * otherStride / 4, 4) { name = "GaussianOtherData" };
            var newSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, newSplatCount * shStride / 4, 4) { name = "GaussianSHData" };
            var newColorData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, newSplatCount * colorStride / 4, 4) { name = "GaussianColorData" };

            // new texture is a RenderTexture so we can write to it from a compute shader
            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(newSplatCount);
            var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
            var newTexColorData = new RenderTexture(texWidth, texHeight, texFormat, GraphicsFormat.None) { name = "GaussianColorData", enableRandomWrite = true };
            newTexColorData.Create();

            // selected/deleted buffers
            var selTarget = GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource | GraphicsBuffer.Target.CopyDestination;
            var selSize = (newSplatCount + 31) / 32;
            var newEditSelected = new GraphicsBuffer(selTarget, selSize, 4) { name = "GaussianSplatSelected" };
            var newEditSelectedMouseDown = new GraphicsBuffer(selTarget, selSize, 4) { name = "GaussianSplatSelectedInit" };
            var newEditDeleted = new GraphicsBuffer(selTarget, selSize, 4) { name = "GaussianSplatDeleted" };
            ClearGraphicsBuffer(newEditSelected);
            ClearGraphicsBuffer(newEditSelectedMouseDown);
            ClearGraphicsBuffer(newEditDeleted);

            var newGpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, newSplatCount, kGpuViewDataSize);
            InitSortBuffers(newSplatCount);

            // copy existing data over into new buffers
            EditCopySplats(transform, newPosData, newOtherData, newSHData, newColorData, newTexColorData, newEditDeleted, newSplatCount, 0, 0, m_SplatCount);

            // use the new buffers and the new splat count
            m_GpuPosData?.Dispose();
            m_GpuOtherData?.Dispose();
            m_GpuSHData?.Dispose();
            m_GpuColorData?.Dispose();
            DestroyImmediate(m_GpuTexColorData);
            m_GpuView?.Dispose();

            m_GpuEditSelected?.Dispose();
            m_GpuEditSelectedMouseDown?.Dispose();
            m_GpuEditDeleted?.Dispose();

            m_GpuPosData = newPosData;
            m_GpuOtherData = newOtherData;
            m_GpuSHData = newSHData;
            m_GpuColorData = newColorData;
            m_GpuTexColorData = newTexColorData;
            m_GpuView = newGpuView;
            m_GpuEditSelected = newEditSelected;
            m_GpuEditSelectedMouseDown = newEditSelectedMouseDown;
            m_GpuEditDeleted = newEditDeleted;

            DisposeBuffer(ref m_GpuEditPosMouseDown);
            DisposeBuffer(ref m_GpuEditOtherMouseDown);

            m_SplatCount = newSplatCount;
            editModified = true;
        }

        public void EditCopySplatsInto(GaussianSplatRenderer dst, int copySrcStartIndex, int copyDstStartIndex, int copyCount)
        {
            EditCopySplats(
                dst.transform,
                dst.m_GpuPosData, dst.m_GpuOtherData, dst.m_GpuSHData, dst.m_GpuColorData, dst.m_GpuTexColorData, dst.m_GpuEditDeleted,
                dst.splatCount,
                copySrcStartIndex, copyDstStartIndex, copyCount);
            dst.editModified = true;
        }

        public void EditCopySplats(
            Transform dstTransform,
            GraphicsBuffer dstPos, GraphicsBuffer dstOther, GraphicsBuffer dstSH, GraphicsBuffer dstColor, Texture dstColorTex,
            GraphicsBuffer dstEditDeleted,
            int dstSize,
            int copySrcStartIndex, int copyDstStartIndex, int copyCount)
        {
            if (!EnsureEditingBuffers()) return;

            Matrix4x4 copyMatrix = dstTransform.worldToLocalMatrix * transform.localToWorldMatrix;
            Quaternion copyRot = copyMatrix.rotation;
            Vector3 copyScale = copyMatrix.lossyScale;

            using var cmb = new CommandBuffer { name = "SplatCopy" };
            SetAssetDataOnCS(cmb, KernelIndices.CopySplats);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstPos", dstPos);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstOther", dstOther);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstSH", dstSH);
            cmb.SetComputeTextureParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstTexColor", dstColorTex);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstColor", dstColor);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstEditDeleted", dstEditDeleted);

            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopyDstSize", dstSize);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopySrcStartIndex", copySrcStartIndex);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopyDstStartIndex", copyDstStartIndex);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopyCount", copyCount);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_CopyTransformRotation", new Vector4(copyRot.x, copyRot.y, copyRot.z, copyRot.w));
            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_CopyTransformScale", copyScale);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, "_CopyTransformMatrix", copyMatrix);

            DispatchUtilsAndExecute(cmb, KernelIndices.CopySplats, copyCount);
        }
        void DispatchUtilsAndExecute(CommandBuffer cmb, KernelIndices kernel, int count)
        {
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)kernel, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)kernel, (int)((count + gsX - 1) / gsX), 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);
        }
        public GraphicsBuffer GpuEditDeleted => m_GpuEditDeleted;
    }
}