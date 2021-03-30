#include <iostream>
#include <string>
#include <cstdio>
extern "C"
{
#include <pci/pci.h>
}

using namespace std;

int main()
{
	int i = 0;
	uint8_t ret;
	uint32_t dom, bus, dev, func;

	struct pci_access *myaccess;
	struct pci_dev *mydev;

	myaccess = pci_alloc();
	pci_init(myaccess);

	mydev = pci_get_dev(myaccess, 0, 1, 0, 0);

	for (i = 0; i < 256; i++) {
		ret = pci_read_byte(mydev, i);
		printf("%d: %02x\n", i, ret);
	}

	if(mydev->phy_slot)
		printf("Phy slot = %c\n", mydev->phy_slot);

	pci_free_dev(mydev);

	pci_cleanup(myaccess);

	std::sscanf("000A:0B:0C.1", "%04x:%02x:%02x.%1u", &dom, &bus, &dev, &func);
	printf("%04x:%02x:%02x.%01x\n", dom, bus, dev, func);

	return 0;
}
