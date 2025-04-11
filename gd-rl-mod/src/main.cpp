#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>

using namespace geode::prelude;

class $modify(MyPlayLayer, PlayLayer) {
	bool init(GJGameLevel* level, bool p1, bool p2) {
		log::info("Level started");
		return PlayLayer::init(level, p1, p2);
	}
	
	void destroyPlayer(PlayerObject *player, GameObject *p1) {
		PlayLayer::destroyPlayer(player, p1);

		if (!player->m_isDead) return;

		log::info("player died at {} percent", (m_player1->getPositionX() / m_levelLength)*100.0f); // there's probably a field for this but idk what it is
	}
};
