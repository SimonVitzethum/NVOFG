// pacing_probe.cpp — headless Vulkan frame-pacing availability probe.
//
// Klaert, womit Frame-Pacing spaeter laeuft (RTX 5070 / Treiber 610.57):
//   1. Instance mit VK_KHR_get_physical_device_properties2, NVIDIA-Device waehlen.
//   2. Device-Extensions pruefen: VK_NV_low_latency2, VK_NV_low_latency,
//      VK_KHR_present_wait, VK_KHR_present_id, VK_GOOGLE_display_timing.
//   3. Falls VK_NV_low_latency2 anwesend: Einstiegspunkte per vkGetDeviceProcAddr
//      laden (nur Verfügbarkeit, KEIN Device, KEIN Sleep-Call).
//
// Return: 0 = low_latency2 komplett da (Ext + alle 5 Entry-Points),
//         2 = nur Teile / Fallback-Lage, 1 = Fehler (keine Instance / kein Device).
// Standalone: g++ -o /tmp/pacing_probe renderfx/src/pacing_probe.cpp -lvulkan
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* kWantedExts[] = {
    "VK_NV_low_latency2",
    "VK_NV_low_latency",
    "VK_KHR_present_wait",
    "VK_KHR_present_id",
    "VK_GOOGLE_display_timing",
};

const char* kLl2EntryPoints[] = {
    "vkSetLatencySleepModeNV",
    "vkLatencySleepNV",
    "vkGetLatencyTimingsNV",
    "vkQueueNotifyOutOfBandNV",
    "vkSetLatencyMarkerNV",
};

bool hasExt(const std::vector<VkExtensionProperties>& list, const char* name) {
    for (const auto& e : list)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

}  // namespace

int main() {
    // 1. Instance mit VK_KHR_get_physical_device_properties2 (Muster: ngx_probe.cpp).
    const char* instExts[] = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "pacing_probe";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = instExts;

    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        std::printf("FEHLER: keine Vulkan-Instance\n");
        return 1;
    }

    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(inst, &n, nullptr) != VK_SUCCESS || n == 0) {
        std::printf("FEHLER: keine PhysicalDevices\n");
        vkDestroyInstance(inst, nullptr);
        return 1;
    }
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());

    VkPhysicalDevice pd = VK_NULL_HANDLE;
    for (auto c : pds) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(c, &p);
        if (p.vendorID != 0x10DE) continue;  // NVIDIA
        pd = c;
        break;
    }
    if (pd == VK_NULL_HANDLE) {
        std::printf("FEHLER: kein NVIDIA-Device gefunden\n");
        vkDestroyInstance(inst, nullptr);
        return 1;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device: %s (vendor 0x%04X, driver 0x%08X, api %u.%u.%u)\n", props.deviceName,
                props.vendorID, props.driverVersion, VK_VERSION_MAJOR(props.apiVersion),
                VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));

    // 2. Device-Extensions abfragen (headless: kein Surface/Swapchain noetig).
    uint32_t ec = 0;
    if (vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, nullptr) != VK_SUCCESS) {
        std::printf("FEHLER: Extension-Query gescheitert\n");
        vkDestroyInstance(inst, nullptr);
        return 1;
    }
    std::vector<VkExtensionProperties> exts(ec);
    if (vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, exts.data()) != VK_SUCCESS) {
        std::printf("FEHLER: Extension-Query gescheitert\n");
        vkDestroyInstance(inst, nullptr);
        return 1;
    }

    std::printf("\nExtension-Tabelle:\n");
    bool present[5] = {};
    for (int i = 0; i < 5; ++i) {
        present[i] = hasExt(exts, kWantedExts[i]);
        std::printf("  %-26s %s\n", kWantedExts[i], present[i] ? "ANWESEND" : "FEHLT");
    }

    // 3. low_latency2-Einstiegspunkte — reine Verfuegbarkeits-Probe:
    // KEIN Device, KEIN Sleep-Call. Primärpfad ist vkGetDeviceProcAddr (wie
    // beauftragt); ohne VkDevice muss dort VK_NULL_HANDLE stehen, was der Loader
    // für Device-Funktionen mit NULL beantwortet — deshalb Gegenprobe per
    // vkGetInstanceProcAddr (liefert dieselben Treiber-Pointer, vgl. Messung:
    // gipa != NULL, gdpa(NULL) == NULL). Pointer werden nie aufgerufen.
    bool ll2ext = present[0];
    bool found[5] = {};
    int foundCount = 0;
    if (ll2ext) {
        auto gdpa = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(inst, "vkGetDeviceProcAddr");
        std::printf("\nEinstiegspunkt-Tabelle (VK_NV_low_latency2):\n");
        if (!gdpa) {
            std::printf("  vkGetDeviceProcAddr selbst FEHLT (alle 5 FEHLT)\n");
            for (int i = 0; i < 5; ++i) std::printf("  %-26s FEHLT\n", kLl2EntryPoints[i]);
        } else {
            for (int i = 0; i < 5; ++i) {
                bool viaGdpa = gdpa(VK_NULL_HANDLE, kLl2EntryPoints[i]) != nullptr;
                bool viaGipa =
                    vkGetInstanceProcAddr(inst, kLl2EntryPoints[i]) != nullptr;
                found[i] = viaGdpa || viaGipa;
                foundCount += found[i] ? 1 : 0;
                std::printf("  %-26s %s%s\n", kLl2EntryPoints[i], found[i] ? "GEFUNDEN" : "FEHLT",
                            found[i] ? (viaGdpa ? " (gdpa)" : " (gipa; gdpa(NULL)=0 ohne Device)")
                                     : " (gdpa+gipa)");
            }
        }
    } else {
        std::printf("\nEinstiegspunkt-Tabelle: entfaellt (VK_NV_low_latency2 FEHLT)\n");
    }

    vkDestroyInstance(inst, nullptr);

    // 4. Kompakt-Ergebnis + Return-Code.
    int rc;
    if (ll2ext && foundCount == 5)
        rc = 0;
    else if (ll2ext || present[1] || present[2] || present[3] || present[4] || foundCount > 0)
        rc = 2;
    else
        rc = 2;  // gar kein Pacing-Anker: ebenfalls "nur Teile" (kein harter Fehler)
    std::printf("\nRESULT: low_latency2_ext=%d entrypoints=%d/5 -> rc=%d\n", ll2ext ? 1 : 0,
                foundCount, rc);
    return rc;
}
