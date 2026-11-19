"""Core images <-> the canonical trainit_interfaces messages (ChannelRequest / ChannelStatus)."""

from trainit_interfaces.msg import ChannelRequest, ChannelStatus

from ..core.contract import RequestImage, StatusImage


def status_to_msg(image: StatusImage, stamp) -> ChannelStatus:
    msg = ChannelStatus()
    msg.header.stamp = stamp
    msg.source_session = image.source_session
    msg.heartbeat = image.heartbeat
    msg.state = image.state
    msg.cycle_requester = image.cycle_requester
    msg.cycle_seq = image.cycle_seq
    msg.cycle_done = image.cycle_done
    msg.reject_requester = image.reject_requester
    msg.reject_seq = image.reject_seq
    msg.reject_code = image.reject_code
    msg.fault_code = image.fault_code
    msg.detail = image.detail
    return msg


def request_from_msg(msg: ChannelRequest) -> RequestImage:
    return RequestImage(source_session=msg.source_session, requester_session=msg.requester_session, seq=msg.seq)
