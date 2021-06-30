#include <iostream>
#include <string>
#include <cstdio>
extern "C"
{
#include <pci/pci.h>
}

using namespace std;

typedef union BDF {
	uint32_t u32;
        struct {
                uint8_t func;
                uint8_t dev;
                uint8_t bus;
                uint8_t dummy;
        } u8;
} BDF_t;

#define BDF_NUM 2

int main()
{
	int i = 0, j = 0;
	uint8_t ret;
	uint32_t dom, bus, dev, func;
	BDF_t bdf[BDF_NUM] = {0x010000, 0x040000};

	struct pci_access *pacc;
	struct pci_dev *mydev;
	bool found = false;

	pacc = pci_alloc();
	pci_init(pacc);
	pci_scan_bus(pacc);

	struct pci_dev *d;

	// Query PCI device's parent
	for (i=0; i<BDF_NUM; ++i) {
		for (d=pacc->devices; d; d=d->next) {
			uint8_t type = pci_read_byte(d, PCI_HEADER_TYPE) & 0x7f;
			uint8_t subordinate = pci_read_byte(d, PCI_SUBORDINATE_BUS);
			if (type != PCI_HEADER_TYPE_BRIDGE || subordinate < bdf[i].u8.bus) {
				continue;
			}

			if (pci_read_byte(d, PCI_SECONDARY_BUS) != bdf[i].u8.bus) {
				continue;
			}

			printf("Parent of %02x:%02x.%1x is %02x:%02x.%1x type = %02x\n",
				bdf[i].u8.bus, bdf[i].u8.dev, bdf[i].u8.func,
				d->bus, d->dev, d->func, type);
		}
	}

	// Dump header value of bdf[0]
	mydev = pci_get_dev(pacc, 0, bdf[0].u8.bus, bdf[0].u8.dev, bdf[0].u8.func);

	printf(" 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\n", ret);
	for (i = 0; i < 256; i++) {
		ret = pci_read_byte(mydev, i);
		printf("%02x ", ret);
		++j;
		if (j % 32 == 0) {
			printf("\n");
		}
	}

	// Get slot id od bdf[0] if possible
	if(mydev->phy_slot)
		printf("Phy slot = %c\n", mydev->phy_slot);

	pci_free_dev(mydev);

	pci_cleanup(pacc);

	std::sscanf("000A:0B:0C.1", "%04x:%02x:%02x.%1u", &dom, &bus, &dev, &func);
	printf("%04x:%02x:%02x.%01x\n", dom, bus, dev, func);

	return 0;
}
