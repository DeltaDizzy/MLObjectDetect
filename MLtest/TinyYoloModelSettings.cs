namespace MLtest
{
    public partial class OnnxModelScorer
    {
        public struct TinyYoloModelSettings
        {
            // for checking Tiny yolo2 Model input and output parameter names,
            // you can use tools like Netron, 
            // which is installed by Visual Studio AI Tool

            // input name
            public const string ModelInput = "image";
            // output name
            public const string ModelOutput = "grid";
        }
    }
}
