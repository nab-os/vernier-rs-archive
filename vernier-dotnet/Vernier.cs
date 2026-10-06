using System;
using System.Runtime.InteropServices;

namespace Vernier
{
    // ── Public types ──────────────────────────────────────────────────────────

    /// <summary>
    /// Detected in-plane pose.
    /// <para><see cref="X"/> and <see cref="Y"/> are in the same physical units
    /// as the <c>period</c> argument passed to the detection call.
    /// <see cref="Theta"/> is in radians.</para>
    /// </summary>
    public readonly struct Pose
    {
        /// <summary>Translation along X.</summary>
        public readonly double X;
        /// <summary>Translation along Y.</summary>
        public readonly double Y;
        /// <summary>In-plane rotation angle in radians.</summary>
        public readonly double Theta;

        internal Pose(double x, double y, double theta) { X = x; Y = y; Theta = theta; }

        /// <inheritdoc/>
        public override string ToString() =>
            $"Pose(X={X:F6}, Y={Y:F6}, Theta={Theta:F6})";
    }

    /// <summary>
    /// CPU- or CUDA-backed Vernier pose detector.
    /// <para>
    /// Create with <see cref="Detector()"/> for the CPU backend or
    /// <see cref="CreateCuda"/> for the CUDA backend.
    /// Reuse across frames — the internal FFT planner caches its plan, so
    /// repeated calls on same-size images are cheaper than constructing a new
    /// <see cref="Detector"/> each time.
    /// </para>
    /// </summary>
    public sealed class Detector : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>Creates a CPU-backed detector.</summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown if the native library could not allocate the detector.
        /// </exception>
        public Detector()
        {
            _handle = Native.vernier_detector_new();
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create CPU detector.");
        }

        private Detector(IntPtr handle) { _handle = handle; }

        /// <summary>
        /// Creates a CUDA-backed detector.
        /// </summary>
        /// <returns>A new <see cref="Detector"/> backed by the GPU.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown if no CUDA device is available or the native library was not
        /// built with CUDA support (rebuild with <c>--features cuda</c>).
        /// </exception>
        public static Detector CreateCuda()
        {
            var handle = Native.vernier_detector_new_cuda();
            if (handle == IntPtr.Zero)
            {
                var msg = Native.LastError() ?? "CUDA detector creation failed.";
                throw new InvalidOperationException(msg);
            }
            return new Detector(handle);
        }

        // ── Detection ─────────────────────────────────────────────────────────

        /// <summary>
        /// Periodic (relative) detection: recovers <c>x</c>, <c>y</c> modulo the
        /// pattern period and the in-image orientation <c>theta</c>.
        /// </summary>
        /// <param name="pixels">
        /// Row-major float array, <c>width × height</c> elements in [0, 1].
        /// </param>
        /// <param name="width">Image width in pixels.</param>
        /// <param name="height">Image height in pixels.</param>
        /// <param name="period">Pattern spatial period in physical units.</param>
        /// <param name="sigma">Bandpass filter half-width in frequency bins.</param>
        /// <param name="minFrequency">
        /// Inner spectral annulus radius for peak search; 0 = no lower limit.
        /// </param>
        /// <param name="maxFrequency">
        /// Outer spectral annulus radius; 0 = no upper limit.
        /// </param>
        /// <param name="smoothingSigma">
        /// Gaussian blur sigma applied to the magnitude spectrum before peak
        /// search; 0 disables blurring.
        /// </param>
        /// <returns>Detected <see cref="Pose"/>.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="pixels"/> holds fewer than
        /// <c>width × height</c> elements or a size is not positive.
        /// </exception>
        /// <exception cref="ObjectDisposedException">
        /// Thrown if the detector has been disposed.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        /// Thrown if detection fails (e.g. no carrier peaks found).
        /// </exception>
        public Pose DetectPeriodic(
            float[] pixels, int width, int height, double period,
            double sigma = 3.0,
            int minFrequency = 0,
            int maxFrequency = 0,
            double smoothingSigma = 0.5)
        {
            ThrowIfDisposed();
            CheckImage(pixels, width, height);
            CheckFrequencies(minFrequency, maxFrequency);
            var raw = Native.vernier_detect_periodic(
                _handle, pixels,
                (nuint)width, (nuint)height,
                period, sigma,
                (nuint)minFrequency, (nuint)maxFrequency,
                smoothingSigma);
            return ToPublicPose(raw, "Periodic detection failed.");
        }

        /// <summary>
        /// Megarena absolute detection: recovers an unambiguous
        /// <c>(x, y, theta)</c> by combining the fine phase measurement with
        /// the LFSR binary code embedded in the Megarena pattern.
        /// </summary>
        /// <param name="pixels">
        /// Row-major float array, <c>width × height</c> elements in [0, 1].
        /// </param>
        /// <param name="width">Image width in pixels.</param>
        /// <param name="height">Image height in pixels.</param>
        /// <param name="physicalPeriod">
        /// Pattern spatial period in micrometres (9.0 for the reference pattern).
        /// </param>
        /// <param name="codeSize">
        /// LFSR order in bits (12 for the reference pattern).
        /// </param>
        /// <param name="sigma">Bandpass filter half-width in frequency bins.</param>
        /// <param name="minFrequency">
        /// Inner spectral annulus radius for peak search; 0 = no lower limit.
        /// </param>
        /// <param name="maxFrequency">
        /// Outer spectral annulus radius; 0 = no upper limit.
        /// </param>
        /// <param name="smoothingSigma">
        /// Gaussian blur sigma applied to the magnitude spectrum before peak search.
        /// </param>
        /// <returns>Detected <see cref="Pose"/>.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="pixels"/> holds fewer than
        /// <c>width × height</c> elements or a size is not positive.
        /// </exception>
        /// <exception cref="ObjectDisposedException">
        /// Thrown if the detector has been disposed.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        /// Thrown if detection or LFSR decode fails.
        /// </exception>
        public Pose DetectMegarena(
            float[] pixels, int width, int height,
            double physicalPeriod, int codeSize,
            double sigma = 3.0,
            int minFrequency = 20,
            int maxFrequency = 500,
            double smoothingSigma = 0.5)
        {
            ThrowIfDisposed();
            CheckImage(pixels, width, height);
            CheckFrequencies(minFrequency, maxFrequency);
            var raw = Native.vernier_detect_megarena(
                _handle, pixels,
                (nuint)width, (nuint)height,
                physicalPeriod, checked((uint)codeSize),
                sigma,
                (nuint)minFrequency, (nuint)maxFrequency,
                smoothingSigma);
            return ToPublicPose(raw, "Megarena detection failed.");
        }

        // ── Disposal ──────────────────────────────────────────────────────────

        /// <summary>Releases the native detector handle.</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                Native.vernier_detector_free(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Detector));
        }

        // The native side reads width × height floats from the array, so a
        // short array or a negative size would read past its end.
        private static void CheckImage(float[] pixels, int width, int height)
        {
            if (pixels is null)
                throw new ArgumentNullException(nameof(pixels));
            if (width <= 0 || height <= 0)
                throw new ArgumentException($"Image size {width}×{height} must be positive.");
            if (pixels.LongLength < (long)width * height)
                throw new ArgumentException(
                    $"pixels holds {pixels.LongLength} elements, fewer than {width}×{height}.",
                    nameof(pixels));
        }

        private static void CheckFrequencies(int minFrequency, int maxFrequency)
        {
            if (minFrequency < 0)
                throw new ArgumentOutOfRangeException(nameof(minFrequency));
            if (maxFrequency < 0)
                throw new ArgumentOutOfRangeException(nameof(maxFrequency));
        }

        private static Pose ToPublicPose(RawPose raw, string fallbackMessage)
        {
            if (raw.Found == 0)
                throw new InvalidOperationException(Native.LastError() ?? fallbackMessage);
            return new Pose(raw.X, raw.Y, raw.Theta);
        }
    }

    // ── Native interop ────────────────────────────────────────────────────────

    /// <summary>
    /// Raw P/Invoke declarations for <c>libvernier_cabi</c>.
    /// The runtime resolves to <c>libvernier_cabi.so</c> on Linux,
    /// <c>vernier_cabi.dll</c> on Windows, and
    /// <c>libvernier_cabi.dylib</c> on macOS.
    /// </summary>
    internal static class Native
    {
        private const string Lib = "vernier_cabi";

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vernier_detector_new();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vernier_detector_new_cuda();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vernier_detector_free(IntPtr det);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern RawPose vernier_detect_periodic(
            IntPtr det,
            [In] float[] pixels,
            nuint width,
            nuint height,
            double period,
            double sigma,
            nuint minFrequency,
            nuint maxFrequency,
            double smoothingSigma);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern RawPose vernier_detect_megarena(
            IntPtr det,
            [In] float[] pixels,
            nuint width,
            nuint height,
            double physicalPeriod,
            uint codeSize,
            double sigma,
            nuint minFrequency,
            nuint maxFrequency,
            double smoothingSigma);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr vernier_last_error();

        /// <summary>
        /// Returns the last error string from the native library, or
        /// <see langword="null"/> if there was no error.
        /// </summary>
        internal static string? LastError()
        {
            var ptr = vernier_last_error();
            return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) : null;
        }
    }

    // ── Internal ABI struct ───────────────────────────────────────────────────

    /// <summary>
    /// Mirror of <c>VernierPose</c> in <c>vernier.h</c>.
    /// Must match the C layout exactly: three doubles then one int32.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct RawPose
    {
        public double X;
        public double Y;
        public double Theta;
        public int Found;
    }
}
