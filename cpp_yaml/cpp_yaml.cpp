#include <iostream>
#include <iomanip>
#include "yaml-cpp/yaml.h"

using namespace std;

int main(int argc,char** argv)
{
    YAML::Node config = YAML::LoadFile("led.yaml");

    //cout << config << endl;
    cout << "enabled_cmd:" << config["enabled_cmd"] << endl;
    cout << "led_bdf:" << endl;
    std::vector<int> vi = config["led_bdf"].as<std::vector<int>>();
    for(auto it = vi.begin(); it != vi.end();++it)
    {
        cout << "\t0x" << setfill('0') << setw(6) << hex << *it << endl;
    }
    cout << "led_reg:" << endl;
    cout << "\tgreen:" << endl;
    cout << "\t\taddr:" << config["led_reg"]["green"]["addr"]<< endl;
    cout << "\t\tmask:" << config["led_reg"]["green"]["mask"]<< endl;
    return 0;
}
