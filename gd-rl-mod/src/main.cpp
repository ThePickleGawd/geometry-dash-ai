#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

using namespace geode::prelude;

class $modify(MyPlayerObject, PlayerObject) {
	bool pushButton(PlayerButton btn) {
		log::info("Jump button received by player!");
		// PlayerObject::lockPlayer();
		this->m_isLocked = !this->m_isLocked;
		return PlayerObject::pushButton(btn);
	}
};
