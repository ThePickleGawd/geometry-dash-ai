#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/CCKeyboardDispatcher.hpp>
#include "controls.hpp"
#include "tcpserver/server.hpp"

using namespace geode::prelude;

$on_mod(Loaded) {
	log::info("Mod loaded, let's setup tcp server");
	tcpserver::start();
}

class $modify(RLPlayerObject, PlayerObject) {
	bool pushButton(PlayerButton btn) {
		log::info("Jump button received by player!");
		// controls::freeze();
		return PlayerObject::pushButton(btn);
	}
};

class $modify(RLPlayLayer, PlayLayer) {
	struct Fields {
		bool frameStepMode = false;
		bool waitingNextFrame = false;
		int framesToStep = 0;
	};



	bool init(GJGameLevel* level, bool p1, bool p2) {
		if (!PlayLayer::init(level, p1, p2))
			return false;
		log::info("Level started");
		return true;
	}

	// void postUpdate(float dt) override {
	// 	PlayLayer::postUpdate(dt);
	// }

	void destroyPlayer(PlayerObject* player, GameObject* p1) override {
		PlayLayer::destroyPlayer(player, p1);

		if (!player->m_isDead)
			return;

		log::info("player died at {} percent", (m_player1->getPositionX() / m_levelLength) * 100.0f);
	}
};

class $modify(RLCCKeyboardDispatcher, CCKeyboardDispatcher) {
	bool dispatchKeyboardMSG(enumKeyCodes key, bool down, bool repeat) {
		// if (down) {
		// 	log::info("key pressed");
		// 	if (key == cocos2d::KEY_O) {
		// 		log::info("Pressed O: stepping 1 frame");
		// 		controls::step(1);
		// 	}
		// }
		return CCKeyboardDispatcher::dispatchKeyboardMSG(key, down, repeat);
	}
};